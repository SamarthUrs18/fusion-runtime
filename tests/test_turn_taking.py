"""
Regression tests for turn-completion timing in PipelineOrchestrator._llm_stage.

The bug these guard against: turn-completion used to be decided reactively,
the instant a new STT result arrived — checking accumulated silence *right
then*. But STT only ever produces a new result once new speech has
accumulated (VAD drops silence upstream before it reaches STT), so that
"silence" reading was really the pause *before* the new speech started, not
a pause *after* the user stopped talking. Every ordinary mid-sentence
breath satisfied it, cutting turns off mid-thought.

The fix makes an independent watcher poll `turn_state.silence_ms`
continuously and fire only once *forward*, real-time silence crosses a
threshold — a short one for confidently-complete-sounding text, a longer
one otherwise. These tests drive `turn_state.silence_ms` directly (bypassing
real VAD/audio) so they're fast and deterministic, with a fake LLM so no
model loading is needed.
"""
import asyncio
import contextlib

import pytest

from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import LLMChunk
from fusion_runtime.engine import LatencyBudget, PipelineMetrics, PipelineOrchestrator
from fusion_runtime.engine.streaming import PartialTranscript
from fusion_runtime.contract import Capabilities, TurnDetector, TurnPrediction
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.conversation import Conversation
from fusion_runtime.telemetry import ListSink, telemetry
from fusion_runtime.vad import TurnState


class FakeLLM:
    """Replies instantly with a canned token, so tests are fast and only
    exercise the turn-timing logic, not real generation. Records what it was
    asked, so tests can assert on the transcript the pipeline built."""

    def __init__(self):
        self.user_messages = []

    async def generate(self, request):

        messages = request.messages
        self.user_messages.append(
            next(m.content for m in reversed(messages) if m.role == "user")
        )
        yield LLMChunk(text="ok")
        yield LLMChunk(finish_reason="stop")


class FakeDetector(TurnDetector):
    """A turn detector model stand-in: a fixed probability, optional delay or failure."""

    def __init__(self, probability=0.5, delay_s=0.0, fail=False, uses_audio=False, uses_history=False):
        from fusion_runtime.contract import ModelSpec

        super().__init__(ModelSpec(stage="turn", runtime="fake", model=""))
        self.probability, self.delay_s, self.fail = probability, delay_s, fail
        self.uses_audio, self.uses_history = uses_audio, uses_history
        self.requests = []

    @property
    def capabilities(self):
        return Capabilities(max_concurrency=4)

    async def load(self):
        pass

    async def predict(self, request):
        self.requests.append(request)
        await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("detector model crashed")
        return TurnPrediction(self.probability)


def make_orchestrator(min_confident_ms: int = 100, min_silence_ms: int = 400, max_silence_ms: int = 900,
                      detector=None) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.min_confident_silence_ms = min_confident_ms
    orch.config.turn_detection.min_silence_ms = min_silence_ms
    orch.config.turn_detection.max_silence_ms = max_silence_ms
    orch.turn_detector = detector  # None: silence only, the default
    orch.llm = FakeLLM()
    return orch


async def stt_once_then_stall(text: str):
    """Yields one STT result, then hangs forever without closing — like a
    live session where the user might still say more later."""
    yield PartialTranscript(text=text, confidence=1.0, latency_ms=0)
    await asyncio.Event().wait()


async def hold_then_grow_silence(turn_state: TurnState, hold_ms: float):
    """Simulates VAD: silence stays at 0 (as if speech were ongoing) for
    `hold_ms`, then grows continuously — a controllable stand-in for real
    frame-level VAD, driven directly instead of through audio."""
    turn_state.vad_active = True
    loop = asyncio.get_event_loop()
    start = loop.time()
    while True:
        elapsed_ms = (loop.time() - start) * 1000
        turn_state.silence_ms = max(0.0, elapsed_ms - hold_ms)
        await asyncio.sleep(0.01)


async def cancel_and_wait(task: asyncio.Task):
    """Cancel a task and actually wait for the cancellation to land before
    doing anything else with what it was operating on — cancelling a task
    driving an async generator's __anext__() and immediately calling
    aclose() on that generator races the cancellation's delivery and
    raises "aclose(): asynchronous generator is already running"."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def next_token_checkpoints(gen, checkpoints):
    """Checks whether `gen.__anext__()` has produced a value by each time
    in `checkpoints` (seconds from now), *without* ever cancelling the
    underlying call — a plain `asyncio.wait_for(gen.__anext__(), ...)`
    that times out cancels that call, which for an async generator ends up
    closing it entirely, making every later `__anext__()` raise
    StopAsyncIteration instead of waiting. `asyncio.shield` keeps the call
    alive across a timed-out checkpoint so a later checkpoint can still
    observe its eventual result.

    Returns (results, task): `results` aligned with `checkpoints` (value or
    None per checkpoint, stopping early once a value arrives), and the
    underlying task (caller should cancel it during cleanup if unused).
    """
    task = asyncio.ensure_future(gen.__anext__())
    results = []
    elapsed = 0.0
    for checkpoint in checkpoints:
        try:
            value = await asyncio.wait_for(asyncio.shield(task), timeout=checkpoint - elapsed)
            results.append(value)
            break
        except asyncio.TimeoutError:
            results.append(None)
        elapsed = checkpoint
    return results, task


async def first_reply_times(orch, text, checkpoints, turn_state=None):
    turn_state = turn_state or TurnState()
    driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))
    gen = orch._llm_stage(stt_once_then_stall(text), "system prompt", LatencyBudget(total_ms=500), PipelineMetrics(),
                          turn_state=turn_state)
    try:
        results, task = await next_token_checkpoints(gen, checkpoints)
        await cancel_and_wait(task)
        return results
    finally:
        driver.cancel()
        await gen.aclose()


class TestTurnTiming:
    async def test_default_waits_the_configured_silence_even_after_punctuation(self):
        """Whisper writes "Hello." for a greeting the user hasn't finished ("hello ... my name is").
        Punctuation must not shorten the wait: that's what cut people off mid-sentence."""
        results = await first_reply_times(make_orchestrator(min_silence_ms=400), "Hello.", [0.25, 0.6])
        assert results[0] is None, "answered before the configured silence, because of a full stop"
        assert results[-1] == "ok"

    async def test_likely_finished_prediction_answers_after_the_short_wait(self):
        detector = FakeDetector(probability=0.95)
        results = await first_reply_times(make_orchestrator(min_confident_ms=100, min_silence_ms=400, detector=detector),
                                          "Book a table for two at seven.", [0.3])
        assert results[-1] == "ok"
        assert detector.requests[0].transcript == "Book a table for two at seven."

    async def test_mid_thought_prediction_waits_longer(self):
        detector = FakeDetector(probability=0.05)
        orch = make_orchestrator(min_silence_ms=400, max_silence_ms=900, detector=detector)
        results = await first_reply_times(orch, "I would like a table for", [0.6, 1.2])
        assert results[0] is None, "a likely mid-thought pause got the normal wait"
        assert results[-1] == "ok"

    async def test_failing_detector_keeps_the_default_wait_and_warns_once(self):
        sink = ListSink()
        telemetry.add_sink(sink)
        try:
            orch = make_orchestrator(min_silence_ms=300, detector=FakeDetector(fail=True))
            results = await first_reply_times(orch, "Hello there", [0.6])
        finally:
            telemetry.remove_sink(sink)
        assert results[-1] == "ok"
        failures = sink.named("turn.detector_failed")
        assert len(failures) == 1 and failures[0].error is not None

    async def test_slow_detector_never_holds_the_turn(self):
        orch = make_orchestrator(min_confident_ms=50, min_silence_ms=300, detector=FakeDetector(0.95, delay_s=5))
        results = await first_reply_times(orch, "Thanks, bye.", [0.2, 0.6])
        assert results[0] is None and results[-1] == "ok"

    async def test_audio_and_history_detectors_get_what_they_declare(self):
        detector = FakeDetector(probability=0.95, uses_audio=True, uses_history=True)
        orch = make_orchestrator(min_confident_ms=50, detector=detector)
        turn_state = TurnState()
        turn_state.speech_audio.extend(b"\x01\x00" * 1600)
        await first_reply_times(orch, "Yes please.", [0.4], turn_state=turn_state)
        request = detector.requests[0]
        assert request.audio == b"\x01\x00" * 1600
        assert request.history == ()
        assert turn_state.speech_audio == bytearray(), "turn audio must be cleared once the turn is answered"

    async def test_does_not_fire_while_speech_is_still_ongoing(self):
        """The actual regression case: text that already looks complete,
        but real (forward) silence never accumulates because the user is
        still talking. Must never fire, no matter how long we wait — this
        is exactly what broke when the old code checked silence only at
        the moment new STT text arrived instead of continuously."""
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        turn_state.vad_active = True
        turn_state.silence_ms = 0.0  # held at 0 throughout — "still speaking"

        gen = orch._llm_stage(
            stt_once_then_stall("Hello there."),
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        )
        try:
            results, task = await next_token_checkpoints(gen, [0.6])  # well past both thresholds
            assert results[-1] is None, "fired even though forward silence never actually accumulated"
        finally:
            await cancel_and_wait(task)
            await gen.aclose()

    async def test_flushes_pending_transcript_when_source_ends(self):
        """A finite audio source (e.g. the test fixtures, or a closing
        connection) should still get its last turn processed, not dropped
        silently just because no more silence-watcher ticks make sense."""
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        turn_state.vad_active = True
        turn_state.silence_ms = 0.0  # never crosses either threshold

        async def finite_stt_stream():
            yield PartialTranscript(text="I did not finish", confidence=1.0, latency_ms=0)

        tokens = [
            tok
            async for tok in orch._llm_stage(
                finite_stt_stream(),
                "system prompt",
                LatencyBudget(total_ms=500),
                PipelineMetrics(),
                turn_state=turn_state,
            )
        ]
        assert [t for t in tokens if t] == ["ok"], "pending transcript should be flushed as a final turn when the source ends"

    async def test_supersedes_rather_than_concatenates_stt_results(self):
        """Each PartialTranscript restates the whole turn so far (see PartialTranscript's
        docstring). Appending them instead of replacing stacks overlapping
        re-transcriptions of the same speech, which is what produced
        transcripts like "but I cannot. but I can't speak to you right. but
        I can't speak to you right now. ..." from one ordinary sentence."""
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        turn_state.vad_active = True
        turn_state.silence_ms = 0.0

        async def growing_stt_stream():
            # Successive re-transcriptions of one utterance, each superseding
            # the last — exactly what the real STT emits as a turn grows.
            yield PartialTranscript(text="I can't", confidence=1.0, latency_ms=0)
            yield PartialTranscript(text="I can't speak", confidence=1.0, latency_ms=0)
            yield PartialTranscript(text="I can't speak to you now.", confidence=1.0, latency_ms=0)

        async for _ in orch._llm_stage(
            growing_stt_stream(),
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        ):
            pass

        assert orch.llm.user_messages == ["I can't speak to you now."], (
            f"expected the latest full transcript, got {orch.llm.user_messages!r}"
        )

    async def test_fires_via_wall_clock_fallback_when_vad_is_unavailable(self):
        """The exact bug this guards against: Silero VAD silently failing
        to load (it did, once — a torch/torchaudio ABI mismatch from an
        unrelated dependency upgrade) left `turn_state.vad_active` False
        forever. The watcher used to require vad_active to do anything at
        all, so nothing ever fired — a permanent, silent hang with a live
        mic and zero output, on an infinite (real websocket) audio stream
        with no "end of source" flush to fall back on. Turn completion
        must still work from wall-clock idle time alone when VAD can't be
        trusted, not just when the source eventually ends."""
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        turn_state.vad_active = False  # VAD unavailable, exactly like the real failure
        turn_state.silence_ms = 0.0    # and therefore never meaningfully updated

        gen = orch._llm_stage(
            stt_once_then_stall("Hello there."),  # never closes — like a live session
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        )
        try:
            results, task = await next_token_checkpoints(gen, [0.6])
            assert results[-1] == "ok", "must still fire via wall-clock fallback when VAD never activates"
        finally:
            await cancel_and_wait(task)
            await gen.aclose()


class TestInterruptedReplies:
    async def test_unspoken_words_of_a_cut_off_reply_never_start_the_next_reply(self):
        """ "Your table is booked for—" (interrupted) must not come back as
        "Your table is booked for Sure, 8 pm works." in the next reply."""
        from fusion_runtime.engine.text import speakable_segments

        orch = make_orchestrator(min_silence_ms=50)
        orch.config.turn_detection.resume_window_ms = 0
        barge_in = BargeInState()
        replies = [["Your", " table", " is", " booked", " for"], ["Sure", ",", " 8", " pm", " works", "."]]
        answered = asyncio.Event()

        class LLM:
            async def generate(self, request):
                words = replies.pop(0)
                for i, word in enumerate(words):
                    if replies and i == 4:
                        barge_in.fire()  # the caller talks over the first reply
                    yield LLMChunk(text=word)
                answered.set()
                yield LLMChunk(finish_reason="stop")

        orch.llm = LLM()
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))

        async def stt():
            yield PartialTranscript(text="Book a table.", confidence=1.0, latency_ms=0)
            await asyncio.sleep(0.3)
            yield PartialTranscript(text="Actually make it 8 pm.", confidence=1.0, latency_ms=0)

        try:
            tokens = orch._llm_stage(stt(), "sys", LatencyBudget(total_ms=500), PipelineMetrics(),
                                     turn_state=turn_state, barge_in=barge_in)
            segments = [segment async for segment in speakable_segments(tokens)]
        finally:
            driver.cancel()
        assert segments == ["Your table is booked", "Sure, 8 pm works."] or segments == ["Sure, 8 pm works."]
        assert not any("booked" in s and "Sure" in s for s in segments)



class TestResumedTurns:
    async def test_speaking_right_after_a_pause_joins_the_previous_turn(self):
        """ "hello" ... pause ... "my name is Priya": the reply to "hello" gets cut off by the
        user continuing, so the LLM must see one message, not two separate turns."""
        orch = make_orchestrator(min_silence_ms=50)
        calls = []
        first_reply_started = asyncio.Event()

        class RecordingLLM:
            async def generate(self, request):
                calls.append([(m.role, m.content) for m in request.messages])
                first_reply_started.set()
                yield LLMChunk(text="Hi!")
                yield LLMChunk(finish_reason="stop")

        orch.llm = RecordingLLM()
        barge_in = BargeInState()
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))

        async def stt():
            yield PartialTranscript(text="Hello.", confidence=1.0, latency_ms=0)
            await first_reply_started.wait()
            await asyncio.sleep(0.05)
            barge_in.fire()  # the user talked over the reply to "hello"
            yield PartialTranscript(text="My name is Priya.", confidence=1.0, latency_ms=0)

        try:
            async for _ in orch._llm_stage(stt(), "sys", LatencyBudget(total_ms=500), PipelineMetrics(),
                                           turn_state=turn_state, barge_in=barge_in, conversation=Conversation("sys")):
                pass
        finally:
            driver.cancel()
        assert calls[1] == [("system", "sys"), ("user", "Hello. My name is Priya.")]

    async def test_a_later_interruption_is_a_new_turn(self):
        orch = make_orchestrator(min_silence_ms=50)
        orch.config.turn_detection.resume_window_ms = 100
        calls = []
        replied = asyncio.Event()

        class RecordingLLM:
            async def generate(self, request):
                calls.append([m.content for m in request.messages])
                replied.set()
                yield LLMChunk(text="Hi!")
                yield LLMChunk(finish_reason="stop")

        orch.llm = RecordingLLM()
        barge_in = BargeInState()
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))

        async def stt():
            yield PartialTranscript(text="Hello.", confidence=1.0, latency_ms=0)
            await replied.wait()
            await asyncio.sleep(0.3)  # well past the resume window
            barge_in.fire()
            yield PartialTranscript(text="What are your hours?", confidence=1.0, latency_ms=0)

        try:
            async for _ in orch._llm_stage(stt(), "sys", LatencyBudget(total_ms=500), PipelineMetrics(),
                                           turn_state=turn_state, barge_in=barge_in, conversation=Conversation("sys")):
                pass
        finally:
            driver.cancel()
        assert calls[1] == ["sys", "Hello.", "Hi!", "What are your hours?"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
