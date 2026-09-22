"""Per-call conversation history: sent to the LLM, trimmed to fit, honest about interruptions."""
from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import LLMChunk
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.engine.conversation import Conversation
from fusion_runtime.engine.streaming import PartialTranscript


def roles(messages):
    return [m.role for m in messages]


def test_first_turn_is_system_and_user():
    conversation = Conversation("be brief")
    assert [(m.role, m.content) for m in conversation.messages_for("hi")] == [("system", "be brief"), ("user", "hi")]


def test_previous_turns_are_included_in_order():
    conversation = Conversation("sys")
    conversation.add_turn("book a table", "For how many people?")
    messages = conversation.messages_for("for two")
    assert [(m.role, m.content) for m in messages] == [
        ("system", "sys"), ("user", "book a table"), ("assistant", "For how many people?"), ("user", "for two"),
    ]
    assert conversation.turns == 1


def test_message_limit_keeps_the_most_recent_exchanges():
    conversation = Conversation("sys", max_messages=4)
    for i in range(5):
        conversation.add_turn(f"q{i}", f"a{i}")
    messages = conversation.messages_for("now")
    assert [m.content for m in messages] == ["sys", "q3", "a3", "q4", "a4", "now"]
    assert len(conversation.history) <= 8  # memory stays bounded on long calls


def test_character_budget_drops_oldest_history_first_and_never_starts_on_a_reply():
    conversation = Conversation("s", max_chars=30)
    conversation.add_turn("first question", "first answer")  # 26 chars
    conversation.add_turn("q2", "a2")
    messages = conversation.messages_for("now")
    assert [m.content for m in messages] == ["s", "q2", "a2", "now"]
    assert roles(messages)[1] == "user"


def test_budget_from_context_leaves_room_for_the_reply():
    conversation = Conversation.for_context("s", n_ctx=2048, reply_tokens=256)
    assert conversation.max_chars == (2048 - 256) * 3


def test_interrupted_reply_is_kept_as_far_as_it_got():
    conversation = Conversation("sys")
    conversation.add_turn("tell me the hours", "We open at nine and", interrupted=True)
    assert conversation.history[-1].content == "We open at nine and"


def test_interruption_before_any_reply_carries_the_words_into_the_next_message():
    conversation = Conversation("sys")
    conversation.add_turn("I want to", "", interrupted=True)
    assert conversation.history == []
    assert [m.content for m in conversation.messages_for("change my order")] == ["sys", "I want to change my order"]
    conversation.add_turn("change my order", "Which order?")
    assert conversation.history[0].content == "I want to change my order"
    assert conversation.messages_for("the last one")[-1].content == "the last one"


class RecordingLLM:
    def __init__(self):
        self.calls = []

    async def generate(self, request):

        messages = request.messages
        self.calls.append([(m.role, m.content) for m in messages])
        yield LLMChunk(text="Sure.")
        yield LLMChunk(finish_reason="stop")


async def test_llm_stage_sends_history_on_later_turns():
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.llm = RecordingLLM()
    conversation = Conversation("sys")

    async def one_turn(text):
        yield PartialTranscript(text=text, confidence=1.0, latency_ms=0)

    for text in ("book a table", "for two"):
        async for _ in orch._llm_stage(one_turn(text), "sys",
                                       conversation=conversation):
            pass

    assert orch.llm.calls[0] == [("system", "sys"), ("user", "book a table")]
    assert orch.llm.calls[1] == [("system", "sys"), ("user", "book a table"), ("assistant", "Sure."), ("user", "for two")]
    assert orch.scheduler("llm").completed == 2 and orch.scheduler("llm").in_flight == 0


def test_retracting_a_turn_carries_the_words_and_drops_the_cut_off_reply():
    conversation = Conversation("sys")
    conversation.add_turn("book a table", "For how many?")
    conversation.add_turn("hello", "Hi! How can I")
    assert conversation.retract_last_turn() == "hello"
    assert conversation.has_carried_text
    messages = conversation.messages_for("my name is Priya")
    assert [m.content for m in messages] == ["sys", "book a table", "For how many?", "hello my name is Priya"]
    assert Conversation("sys").retract_last_turn() is None
