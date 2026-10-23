"""The agent speaks first when it has a greeting: said at once, interruptible, remembered."""
import pytest
from fusion_runtime.agent import Agent, AgentError, greeting_for
from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import LLMChunk
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.conversation import Conversation
from fusion_runtime.engine.orchestrator import END_OF_REPLY, REPLY_CUT_OFF
from fusion_runtime.engine.streaming import PartialTranscript

GREETING = "Hi, this is ShopKart. How can I help with your order?"


class RecordingLLM:
    def __init__(self):
        self.calls = []

    async def generate(self, request):
        self.calls.append([(m.role, m.content) for m in request.messages])
        yield LLMChunk(text="It ships Friday.")
        yield LLMChunk(finish_reason="stop")


def orchestrator():
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.llm = RecordingLLM()
    return orch


async def nothing_said():
    return
    yield


async def one_turn(text):
    yield PartialTranscript(text=text, confidence=1.0, latency_ms=0)


async def test_the_greeting_is_spoken_before_the_caller_says_anything():
    events = []
    out = [token async for token in orchestrator()._llm_stage(nothing_said(), "sys", emit=events.append,
                                                              greeting=GREETING)]
    assert out == ["Hi, this is ShopKart.", " How can I help with your order?", END_OF_REPLY]
    assert {"type": "response", "text": GREETING, "is_final": True, "greeting": True} in events


async def test_the_model_knows_it_already_said_hello_without_an_assistant_first_message():
    """Mistral's and older Llamas' templates refuse a conversation that starts with the assistant."""
    orch, conversation = orchestrator(), Conversation("You are ShopKart's order line.")
    async for _ in orch._llm_stage(one_turn("where is order 1042"), "sys", conversation=conversation,
                                   greeting=GREETING):
        pass
    sent = orch.llm.calls[0]
    assert [role for role, _ in sent] == ["system", "user"]
    assert GREETING in sent[0][1] and "Don't greet the caller again" in sent[0][1]


async def test_talking_over_the_greeting_stops_it_and_the_next_reply_is_not_cut():
    barge_in = BargeInState()
    gen = orchestrator()._llm_stage(one_turn("where is my order"), "sys", barge_in=barge_in, greeting=GREETING)
    assert await gen.__anext__() == "Hi, this is ShopKart."
    barge_in.fire()  # the caller starts talking
    rest = [token async for token in gen]
    assert rest[0] == REPLY_CUT_OFF and "How can I help" not in "".join(t for t in rest if isinstance(t, str))
    assert "It ships Friday." in rest  # their question still gets a full answer


async def test_no_greeting_means_the_agent_waits():
    out = [token async for token in orchestrator()._llm_stage(nothing_said(), "sys")]
    assert out == []


def test_a_greeting_is_checked_and_the_environment_can_change_it():
    agent = Agent(prompt="p", greeting="  Hello there.  ")
    assert agent.greeting == "Hello there." and agent.describe()["greeting_chars"] == 12
    assert greeting_for(agent, {}) == "Hello there."
    assert greeting_for(agent, {"FUSION_GREETING": "Namaste, ShopKart here."}) == "Namaste, ShopKart here."
    assert greeting_for(None, {}) is None and greeting_for(Agent(prompt="p"), {"FUSION_GREETING": " "}) is None
    with pytest.raises(AgentError, match="greeting must be the words"):
        Agent(prompt="p", greeting="   ")
    with pytest.raises(AgentError, match="keep it under 400"):
        Agent(prompt="p", greeting="Hello. " * 100)


# ---- said at once: made when the server starts, not on each call -------------------------------

class CountingTTS:
    def __init__(self):
        self.said = []

    async def synthesize(self, request):
        from fusion_runtime.contract.tts import AudioChunk

        self.said.append(request.text)
        yield AudioChunk(pcm=bytes([len(self.said)]) * 480, sample_rate=24000)


async def test_the_greeting_is_made_once_and_every_call_plays_it_without_waiting():
    """Phones notice a pause at the start of a call most: the greeting is synthesized at startup."""
    orch = orchestrator()
    orch.tts = CountingTTS()
    audio_s = await orch.prerender(GREETING)
    assert orch.tts.said == ["Hi, this is ShopKart.", "How can I help with your order?"] and audio_s == 0.02

    async def greeting_tokens():
        async for token in orch._llm_stage(nothing_said(), "sys", greeting=GREETING):
            yield token

    played = [pcm async for pcm in orch._tts_stage(greeting_tokens())]
    assert len(orch.tts.said) == 2  # nothing synthesized again
    assert played == [bytes([1]) * 480, bytes([2]) * 480]  # the stored audio, in order


async def test_a_different_line_is_still_synthesized():
    orch = orchestrator()
    orch.tts = CountingTTS()
    await orch.prerender(GREETING)

    async def reply():
        yield "It ships Friday."
        yield END_OF_REPLY

    assert [pcm async for pcm in orch._tts_stage(reply())] == [bytes([3]) * 480]
    assert orch.tts.said[-1] == "It ships Friday."
