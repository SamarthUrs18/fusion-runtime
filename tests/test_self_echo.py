"""
Regression tests for text-domain self-echo rejection.

The bug: the client mutes its mic while the bot talks and only unmutes on
a genuine (or falsely-triggered) barge-in, or once TTS playback has fully
finished. Either way there's a brief window — room reverb decaying after
`_hard_stop_playback()`, or before playback state has fully settled — where
the bot's *own* voice can still reach the mic. If that gets transcribed, it
looks exactly like a new user turn, and the pipeline dutifully replies to
its own words, which is the "reading its own words" failure seen live.

The device echo canceller (fusion_runtime/echo_canceller.py) removes most of
that before any audio is sent. This check is the backstop for whatever slips
through, and it needs no audio at all: we already know the bot's own text
with zero noise, because we generated it ourselves. So this compares the
candidate "user" text against the bot's own most recently spoken text and
discards it if a long enough run of words matches — see
`PipelineOrchestrator._looks_like_self_echo` and
`TurnDetectionConfig.echo_min_match_words/echo_containment_ratio`.
"""
import asyncio
import contextlib

import pytest

from fusion_runtime.config import PipelineConfig, TurnDetectionConfig
from fusion_runtime.llm import LLMResult
from fusion_runtime.engine import LatencyBudget, PipelineMetrics, PipelineOrchestrator
from fusion_runtime.stt import STTResult
from fusion_runtime.vad import PunctuationTurnDetector, TurnState


def make_orchestrator(min_confident_ms: int = 50, min_silence_ms: int = 400) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.min_confident_silence_ms = min_confident_ms
    orch.config.turn_detection.min_silence_ms = min_silence_ms
    orch.turn_detector = PunctuationTurnDetector(orch.config.turn_detection)
    return orch


async def hold_then_grow_silence(turn_state: TurnState, hold_ms: float = 0.0):
    """Same driver used in test_turn_taking.py: simulates VAD-measured
    forward silence growing continuously, so confident-complete text fires
    quickly and deterministically without depending on real timing."""
    turn_state.vad_active = True
    loop = asyncio.get_event_loop()
    start = loop.time()
    while True:
        elapsed_ms = (loop.time() - start) * 1000
        turn_state.silence_ms = max(0.0, elapsed_ms - hold_ms)
        await asyncio.sleep(0.01)


async def cancel_and_wait(task: asyncio.Task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def next_token_checkpoints(gen, checkpoints):
    """See test_turn_taking.py for why this needs asyncio.shield rather
    than a plain wait_for."""
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


class TestLooksLikeSelfEcho:
    """Unit tests for the matcher itself, independent of the pipeline."""

    def test_flags_near_verbatim_echo(self):
        # Real example from a live session: the bot's own sentence,
        # transcribed back with the trailing word truncated.
        bot_text = "I'm just a voice assistant and can't provide emotional support."
        candidate = "voice assistant and can't provide emotion."
        assert PipelineOrchestrator._looks_like_self_echo(candidate, bot_text, TurnDetectionConfig())

    def test_flags_echo_even_with_stt_mistranscription_layered_on_top(self):
        # Real example: real echo, but Whisper also hallucinated/misheard a
        # few words on top of it ("product" -> "program", "by a group of
        # experts or consumers" -> "installed by group of X"). The matcher
        # only needs one long contiguous run to still be intact.
        bot_text = ("The ratings are the official evaluations of an entity or product "
                    "by a group of experts or consumers.")
        candidate = "the official evaluations of an entity or program.  Again, installed by group of X."
        assert PipelineOrchestrator._looks_like_self_echo(candidate, bot_text, TurnDetectionConfig())

    def test_does_not_flag_genuine_short_topic_overlap(self):
        # A real follow-up question naturally reuses a couple of the bot's
        # own words (ordinary topic continuity) — must not be treated as
        # an echo just because "computer" appears in both.
        bot_text = "A computer is a device that can process information and perform calculations."
        candidate = "what about a computer's memory?"
        assert not PipelineOrchestrator._looks_like_self_echo(candidate, bot_text, TurnDetectionConfig())

    def test_does_not_flag_a_short_reply_that_happens_to_be_a_substring(self):
        # A single short word ("yes") appearing somewhere in a long bot
        # reply must never be enough on its own — see
        # echo_min_match_words.
        bot_text = "A computer is a device that can process information."
        assert not PipelineOrchestrator._looks_like_self_echo("yes", bot_text, TurnDetectionConfig())

    def test_no_prior_bot_text_never_flags(self):
        assert not PipelineOrchestrator._looks_like_self_echo("tell me about computer.", "", TurnDetectionConfig())
        assert not PipelineOrchestrator._looks_like_self_echo("", "some bot text here", TurnDetectionConfig())


class TestPipelineDiscardsSelfEcho:
    """Integration-ish test through _llm_stage: a candidate turn that's
    really just the bot's own last reply bleeding back through the mic
    must never reach the LLM."""

    async def test_does_not_reply_to_its_own_echo(self):
        bot_reply = "A computer is a device that can process information and perform calculations."

        class VerboseFakeLLM:
            def __init__(self):
                self.user_messages = []

            async def generate_stream(self, messages, budget_ms=None):
                self.user_messages.append(
                    next(m.content for m in reversed(messages) if m.role == "user")
                )
                yield LLMResult(text=bot_reply, is_final=False, tokens_used=1, latency_ms=0)
                yield LLMResult(text="", is_final=True, tokens_used=1, latency_ms=0)

        orch = make_orchestrator(min_confident_ms=50, min_silence_ms=400)
        orch.llm = VerboseFakeLLM()
        turn_state = TurnState()
        driver = asyncio.create_task(hold_then_grow_silence(turn_state))

        async def stt_stream():
            yield STTResult(text="tell me about computer.", is_final=True, confidence=1.0, latency_ms=0)
            await asyncio.sleep(0.3)
            # Speaker bleed of `bot_reply` above, transcribed as if it were
            # a brand new turn — a long contiguous run of the bot's own
            # words, exactly the shape seen live.
            yield STTResult(
                text="a device that can process information and perform.",
                is_final=True, confidence=1.0, latency_ms=0,
            )
            await asyncio.Event().wait()  # keep the stream open, like a live session

        gen = orch._llm_stage(
            stt_stream(),
            "system prompt",
            LatencyBudget(total_ms=500),
            PipelineMetrics(),
            turn_state=turn_state,
        )
        task1 = task2 = None
        try:
            first_turn, task1 = await next_token_checkpoints(gen, [0.3])
            assert first_turn[-1] == bot_reply, "the genuine first turn should still get a real reply"

            second_turn, task2 = await next_token_checkpoints(gen, [1.0])
            assert second_turn[-1] is None, "no token should be produced for a discarded echo turn"
            assert orch.llm.user_messages == ["tell me about computer."], (
                f"the bot's own echo reached the LLM as if it were a new turn: {orch.llm.user_messages!r}"
            )
        finally:
            if task1 is not None:
                await cancel_and_wait(task1)
            if task2 is not None:
                await cancel_and_wait(task2)
            driver.cancel()
            await gen.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
