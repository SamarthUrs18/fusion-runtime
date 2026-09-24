"""Tool calling: functions as tools, the OpenAI wire format, and the engine's call-and-answer loop."""
import asyncio
import enum
import json
import threading
from typing import List, Literal, Optional

import httpx
import pytest
from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import LLMChunk, LLMRequest, Message, ModelSpec, ToolCall
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.engine.barge_in import BargeInState
from fusion_runtime.engine.conversation import Conversation
from fusion_runtime.engine.streaming import PartialTranscript
from fusion_runtime.engine.text import END_OF_REPLY, REPLY_CUT_OFF
from fusion_runtime.runtimes.openai_http.llm import OpenAIHTTPLLM
from fusion_runtime.tools import MAX_RESULT_CHARS, Tool, ToolError, as_tools, tool

URL = "http://llm.test/v1"


# ---- functions as tools ------------------------------------------------------------------------

class Size(enum.Enum):
    SMALL = "small"
    LARGE = "large"


@tool
async def order_status(order_id: str, detail: Literal["short", "long"] = "short", size: Optional[Size] = None,
                       items: List[int] = ()) -> dict:  # noqa: B006 - never mutated
    """Look up where an order is.

    Only for orders placed in the last year.

    Args:
        order_id: The order number,
            as the caller reads it out.
        detail (str): How much to say.
    """
    return {"id": order_id, "size": size.value if size else None, "detail": detail}


def test_schema_comes_from_type_hints_and_the_docstring():
    assert isinstance(order_status, Tool)
    assert order_status.description == "Look up where an order is."
    params = order_status.parameters
    assert params["required"] == ["order_id"]
    assert params["properties"]["order_id"] == {"type": "string",
                                                "description": "The order number, as the caller reads it out."}
    assert params["properties"]["detail"] == {"enum": ["short", "long"], "type": "string",
                                              "description": "How much to say.", "default": "short"}
    assert params["properties"]["size"]["enum"] == ["small", "large"]
    assert params["properties"]["items"]["items"] == {"type": "integer"}
    assert order_status.spec.name == "order_status"


async def test_arguments_are_checked_and_enums_arrive_as_members():
    ok = await order_status.run(json.dumps({"order_id": "A1", "size": "large"}))
    assert ok.ok and json.loads(ok.content) == {"id": "A1", "size": "large", "detail": "short"}

    for arguments, message in (("not json", "valid JSON"), ("[1]", "JSON object"), ("{}", "needs order_id"),
                               ('{"order_id": "A1", "orderid": 2}', "no parameter orderid"),
                               ('{"order_id": "A1", "size": "medium"}', "one of 'small', 'large'")):
        result = await order_status.run(arguments)
        assert not result.ok and result.error == "bad_arguments" and message in result.content


async def test_failures_and_timeouts_become_errors_the_model_can_read():
    @tool
    async def broken() -> str:
        """Always fails."""
        raise KeyError("no such order")

    @tool(timeout_s=0.05)
    async def slow() -> str:
        """Takes too long."""
        await asyncio.sleep(1)

    failed = await broken.run("{}")
    assert failed.error == "failed" and "KeyError" in failed.content
    timed_out = await slow.run("")
    assert timed_out.error == "timeout" and "0.05 s" in timed_out.content


async def test_ordinary_functions_run_off_the_event_loop():
    loop_thread = threading.get_ident()

    def where() -> str:
        """Which thread am I on."""
        return "worker" if threading.get_ident() != loop_thread else "loop"

    [where_tool] = as_tools([where])
    assert (await where_tool.run("{}")).content == "worker"


async def test_results_are_text_and_long_ones_are_cut():
    def nothing() -> None:
        """Returns nothing."""

    def huge() -> str:
        """Returns too much."""
        return "x" * (MAX_RESULT_CHARS + 10)

    nothing_tool, huge_tool = as_tools([nothing, huge])
    assert (await nothing_tool.run("{}")).content == "done"
    assert len((await huge_tool.run("{}")).content) < MAX_RESULT_CHARS + 50


def test_definitions_that_cant_work_are_refused():
    def undocumented(x: int) -> int:
        return x

    class Custom:
        pass

    def custom(value: Custom) -> str:
        """Takes something the model can't produce."""
        return ""

    with pytest.raises(ToolError, match="needs a description"):
        tool(undocumented)
    assert tool(undocumented, description="Doubles").description == "Doubles"
    with pytest.raises(ToolError, match="can't be described"):
        tool(custom)
    with pytest.raises(ToolError, match="tool name"):
        tool(undocumented, name="has spaces", description="x")
    with pytest.raises(ToolError, match="timeout_s"):
        tool(undocumented, description="x", timeout_s=0)
    with pytest.raises(ToolError, match="list of functions"):
        as_tools(order_status)  # type: ignore[arg-type]


def test_a_tool_is_still_the_plain_function():
    @tool
    def add(a: int, b: int = 1) -> int:
        """Adds."""
        return a + b

    assert add(2, b=3) == 5


# ---- openai_http: the wire format ----------------------------------------------------------

class ToolServer:
    """Streams a scripted list of SSE `choices[0]` payloads, and records request bodies."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.bodies = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        self.bodies.append(json.loads(request.content))
        script = self.scripts.pop(0)

        async def events():
            for choice in script:
                yield f"data: {json.dumps({'choices': [choice]})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())


def http_runtime(server: ToolServer, **options) -> OpenAIHTTPLLM:
    rt = OpenAIHTTPLLM(ModelSpec(stage="llm", runtime="openai_http", model=URL, options={"model_name": "m", **options}))
    rt.transport = httpx.MockTransport(server.handler)
    return rt


def delta(finish=None, **fields):
    return {"delta": fields, "finish_reason": finish}


async def test_tool_calls_stream_in_pieces_and_arrive_whole():
    server = ToolServer([
        delta(content="Let me check."),
        delta(tool_calls=[{"index": 0, "id": "call_a", "type": "function",
                           "function": {"name": "order_status", "arguments": ""}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"order_'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": 'id": "A1"}'}},
                          {"index": 1, "id": "call_b", "function": {"name": "hours", "arguments": {}}}]),
        delta(finish="tool_calls"),
    ])
    rt = http_runtime(server, extra_body={"top_k": 20})
    await rt.load()
    assert rt.capabilities.tools
    chunks = [c async for c in rt.generate(LLMRequest(messages=[Message("user", "where is A1")],
                                                      tools=(order_status.spec,)))]
    assert chunks[0].text == "Let me check."
    last = chunks[-1]
    assert last.finish_reason == "tool_calls"
    assert last.tool_calls == (ToolCall("call_a", "order_status", '{"order_id": "A1"}'), ToolCall("call_b", "hours", "{}"))
    body = server.bodies[0]
    assert body["tools"] == [{"type": "function", "function": {
        "name": "order_status", "description": order_status.description, "parameters": order_status.parameters}}]
    assert body["tool_choice"] == "auto" and body["top_k"] == 20
    await rt.close()


async def test_calls_finished_with_stop_still_count_and_plain_replies_send_no_tools():
    server = ToolServer(
        [delta(tool_calls=[{"index": 0, "function": {"name": "hours", "arguments": "{}"}}]), delta(finish="stop")],
        [delta(content="Hi."), delta(finish="stop")],
    )
    rt = http_runtime(server)
    await rt.load()
    first = [c async for c in rt.generate(LLMRequest(messages=[Message("user", "hours?")],
                                                     tools=(order_status.spec,)))]
    assert first[-1].finish_reason == "tool_calls" and first[-1].tool_calls[0].id == "call_0"
    second = [c async for c in rt.generate(LLMRequest(messages=[Message("user", "hi")]))]
    assert second[-1].finish_reason == "stop" and not second[-1].tool_calls
    assert "tools" not in server.bodies[1]
    await rt.close()


async def test_tool_exchanges_are_sent_back_in_the_openai_format():
    server = ToolServer([delta(content="Shipped."), delta(finish="stop")])
    rt = http_runtime(server)
    await rt.load()
    call = ToolCall("call_a", "order_status", '{"order_id": "A1"}')
    messages = [Message("user", "where is A1"), Message("assistant", "", tool_calls=(call,)),
                Message("tool", '{"status": "shipped"}', name="order_status", tool_call_id="call_a")]
    [c async for c in rt.generate(LLMRequest(messages=messages))]
    sent = server.bodies[0]["messages"]
    assert sent[1] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_a", "type": "function", "function": {"name": "order_status", "arguments": '{"order_id": "A1"}'}}]}
    assert sent[2] == {"role": "tool", "content": '{"status": "shipped"}', "name": "order_status",
                       "tool_call_id": "call_a"}
    await rt.close()


# ---- the engine: call, answer, and interruptions ---------------------------------------------

class ScriptedLLM:
    """Replies from a script; each reply is a list of chunks. Records what each request carried."""

    def __init__(self, *replies, tools=True):
        self.replies = list(replies)
        self.requests: List[LLMRequest] = []
        self.supports_tools = tools

    @property
    def capabilities(self):
        from fusion_runtime.contract import Capabilities
        return Capabilities(tools=self.supports_tools, max_concurrency=4)

    async def generate(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0) if self.replies else [LLMChunk(text="Done.", finish_reason="stop")]
        for chunk in reply:
            yield chunk


def calls(*names):
    return LLMChunk(finish_reason="tool_calls",
                    tool_calls=tuple(ToolCall(f"call_{i}", name, "{}") for i, name in enumerate(names)))


def orchestrator(llm, **llm_settings) -> PipelineOrchestrator:
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    config = PipelineConfig()
    orch.config = config.model_copy(update={"llm": config.llm.model_copy(update=llm_settings)})
    orch.llm = llm
    return orch


async def one_turn(text):
    yield PartialTranscript(text=text, confidence=1.0, latency_ms=0)


def order_tools(log):
    @tool
    async def lookup(order_id: str = "A1") -> dict:
        """Look up an order."""
        log.append("lookup")
        return {"status": "shipped"}

    return as_tools([lookup])


async def test_the_model_calls_a_tool_then_answers_with_its_result():
    log = []
    llm = ScriptedLLM(
        [LLMChunk(text="Let me check."), calls("lookup")],
        [LLMChunk(text="It shipped."), LLMChunk(finish_reason="stop")],
    )
    orch = orchestrator(llm)
    conversation = Conversation("sys")
    events = []
    tokens = [t async for t in orch._llm_stage(one_turn("where is my order"), "sys", emit=events.append,
                                                conversation=conversation, tools=order_tools(log))]

    assert log == ["lookup"]
    # "Let me check." is flushed to TTS before the tool runs, so the caller hears it while waiting
    assert tokens == ["Let me check.", END_OF_REPLY, "It shipped.", END_OF_REPLY]
    first, second = llm.requests
    assert [t.name for t in first.tools] == ["lookup"]
    sent = second.messages
    assert sent[-2].role == "assistant" and sent[-2].content == "Let me check."
    assert sent[-2].tool_calls[0].name == "lookup"
    assert sent[-1].role == "tool" and sent[-1].tool_call_id == "call_0" and "shipped" in sent[-1].content
    # One final response event, with the whole reply; the history keeps what was said, not the tool traffic
    finals = [e for e in events if e.get("type") == "response" and e["is_final"]]
    assert len(finals) == 1 and finals[0]["text"] == "Let me check. It shipped."
    assert [(m.role, m.content) for m in conversation.history] == [
        ("user", "where is my order"), ("assistant", "Let me check. It shipped.")]
    assert orch.scheduler("llm").in_flight == 0


async def test_unknown_tools_and_failures_are_reported_to_the_model_not_raised():
    llm = ScriptedLLM([calls("refund")], [LLMChunk(text="Sorry.", finish_reason="stop")])
    orch = orchestrator(llm)
    tokens = [t async for t in orch._llm_stage(one_turn("refund me"), "sys", conversation=Conversation("sys"),
                                                tools=order_tools([]))]
    assert "Sorry." in tokens
    tool_message = llm.requests[1].messages[-1]
    assert tool_message.role == "tool" and "no tool named 'refund'" in tool_message.content


async def test_rounds_are_capped_and_the_last_one_offers_no_tools():
    log = []
    llm = ScriptedLLM([calls("lookup")], [calls("lookup")], [LLMChunk(text="Here.", finish_reason="stop")])
    orch = orchestrator(llm, max_tool_rounds=2)
    tokens = [t async for t in orch._llm_stage(one_turn("again"), "sys", conversation=Conversation("sys"),
                                                tools=order_tools(log))]
    assert log == ["lookup", "lookup"]
    assert [bool(r.tools) for r in llm.requests] == [True, True, False]
    assert tokens[-2:] == ["Here.", END_OF_REPLY]


async def test_without_tools_nothing_changes():
    llm = ScriptedLLM([LLMChunk(text="Hi."), LLMChunk(finish_reason="stop")])
    orch = orchestrator(llm)
    tokens = [t async for t in orch._llm_stage(one_turn("hello"), "sys", conversation=Conversation("sys"))]
    assert tokens == ["Hi.", END_OF_REPLY]
    assert llm.requests[0].tools == ()


async def test_talking_over_a_slow_tool_cancels_it_and_drops_the_answer():
    started, cancelled = asyncio.Event(), asyncio.Event()

    @tool(timeout_s=5)
    async def slow_lookup() -> str:
        """A slow backend."""
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "late"

    llm = ScriptedLLM([LLMChunk(text="One moment."), calls("slow_lookup")])
    orch = orchestrator(llm)
    barge_in = BargeInState()
    conversation = Conversation("sys")
    events = []

    async def interrupt():
        await started.wait()
        barge_in.fire()

    interrupter = asyncio.create_task(interrupt())
    tokens = [t async for t in orch._llm_stage(one_turn("check it"), "sys", emit=events.append, barge_in=barge_in,
                                                conversation=conversation, tools=(slow_lookup,))]
    await interrupter
    assert cancelled.is_set()
    assert tokens[-1] == REPLY_CUT_OFF
    assert len(llm.requests) == 1  # never asked the model to answer after the interruption
    assert events[-1]["interrupted"] and events[-1]["text"] == "One moment."
    assert conversation.history[-1].content == "One moment."


def test_a_runtime_without_tool_support_is_refused_at_startup():
    orch = orchestrator(ScriptedLLM(tools=False))
    orch.check_tools(())  # no tools: fine
    with pytest.raises(ValueError, match="can't call them"):
        orch.check_tools(order_tools([]))
    orchestrator(ScriptedLLM()).check_tools(order_tools([]))
