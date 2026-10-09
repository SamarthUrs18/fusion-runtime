"""Interruptions can be switched off: for the greeting ("this call may be recorded"), or for the whole agent."""
import asyncio

from fusion_runtime.agent import Agent, Turns
from fusion_runtime.config import PipelineConfig
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.orchestrator import END_OF_REPLY
from fusion_runtime.engine.streaming import PartialTranscript

GREETING = "This call may be recorded. How can I help?"


def test_protected_speech_cant_be_interrupted_until_it_has_played():
    state = BargeInState()
    state.mark_speaking()
    state.protect()
    assert state.speaking and not state.can_interrupt
    state.mark_idle()
    state.set_playing(True)  # the greeting is still coming out of the speaker
    assert not state.can_interrupt
    state.set_playing(False)  # heard in full
    assert not state.protected
    state.mark_speaking()  # the next reply is interruptible as usual
    assert state.can_interrupt


def test_the_next_reply_ends_protection_even_without_playback_reports():
    state = BargeInState()
    state.mark_speaking()
    state.protect()
    state.mark_idle()
    state.mark_speaking()
    assert state.can_interrupt


def test_turns_interruptible_reaches_the_config():
    assert Agent(prompt="p").config({}).turn_detection.interruptible is True
    config = Agent(prompt="p", turns=Turns(interruptible=False)).config({})
    assert config.turn_detection.interruptible is False


async def test_a_never_interruptible_agent_doesnt_watch_for_interruptions():
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.interruptible = False

    async def endless_audio():
        while True:
            yield b"\x01\x00" * 512
            await asyncio.sleep(0)

    state = BargeInState()
    state.mark_speaking()
    await asyncio.wait_for(orch._barge_in_watcher(endless_audio(), state), timeout=1)  # returns at once
    assert not state.interrupted.is_set()


class QuietLLM:
    async def generate(self, request):
        from fusion_runtime.contract import LLMChunk
        yield LLMChunk(text="Sure.")
        yield LLMChunk(finish_reason="stop")


async def test_a_protected_greeting_is_spoken_in_full_and_the_caller_answered_after():
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.llm = QuietLLM()
    state = BargeInState()

    async def caller_speaks_over_it():
        yield PartialTranscript(text="where is my order", confidence=1.0, latency_ms=0)

    gen = orch._llm_stage(caller_speaks_over_it(), "sys", barge_in=state, greeting=GREETING,
                          greeting_interruptible=False)
    first = await gen.__anext__()
    assert first == "This call may be recorded." and not state.can_interrupt
    rest = [token async for token in gen]
    assert rest[:2] == [" How can I help?", END_OF_REPLY]  # the whole greeting
    assert "Sure." in rest  # and the caller still gets an answer
