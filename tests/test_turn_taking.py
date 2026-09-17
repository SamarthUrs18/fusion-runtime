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
from fusion_runtime.vad import PunctuationTurnDetector, TurnState


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


def make_orchestrator(min_confident_ms: int = 100, min_silence_ms: int = 400) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.min_confident_silence_ms = min_confident_ms
    orch.config.turn_detection.min_silence_ms = min_silence_ms
    orch.turn_detector = PunctuationTurnDetector(orch.config.turn_detection)
    orch.llm = FakeLLM()
    return orch


async def stt_once_then_stall(text: str, is_final: bool = True):
    """Yields one STT result, then hangs forever without closing — like a
    live session where the user might still say more later."""
    yield PartialTranscript(text=text, is_final=is_final, confidence=1.0, latency_ms=0)
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


class TestTurnTiming:
    async def test_confident_text_fires_after_short_silence(self):
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))

        gen = orch._llm_stage(
            stt_once_then_stall("Thanks, bye."),  # ends in punctuation -> "complete"
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        )
        try:
            # Should fire well before min_silence_ms (400ms) — it only
            # needed the short confident-completion threshold (100ms).
            results, task = await next_token_checkpoints(gen, [0.3])
            assert results[-1] == "ok", "confidently-complete text should respond on the short threshold"
        finally:
            await cancel_and_wait(task)
            driver.cancel()
            await gen.aclose()

    async def test_ambiguous_text_waits_for_long_silence(self):
        orch = make_orchestrator(min_confident_ms=100, min_silence_ms=400)
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state, hold_ms=0))

        gen = orch._llm_stage(
            stt_once_then_stall("I did"),  # no terminal punctuation -> "ambiguous"
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        )
        try:
            # Must not fire on the short (confident) threshold (0.2s)...
            # but must eventually fire once the long one (0.4s) is crossed.
            results, task = await next_token_checkpoints(gen, [0.2, 0.5])
            assert results[0] is None, "ambiguous text fired on the short threshold, not the long one"
            assert results[-1] == "ok", "ambiguous text never completed even after the long silence threshold"
        finally:
            await cancel_and_wait(task)
            driver.cancel()
            await gen.aclose()

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
            yield PartialTranscript(text="I did not finish", is_final=False, confidence=1.0, latency_ms=0)

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
        assert tokens == ["ok"], "pending transcript should be flushed as a final turn when the source ends"

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
            yield PartialTranscript(text="I can't", is_final=False, confidence=1.0, latency_ms=0)
            yield PartialTranscript(text="I can't speak", is_final=False, confidence=1.0, latency_ms=0)
            yield PartialTranscript(text="I can't speak to you now.", is_final=True, confidence=1.0, latency_ms=0)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
