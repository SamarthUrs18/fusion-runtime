#!/usr/bin/env python3
"""A voice agent that looks things up while the caller waits. Run it with:

    vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8001 --enable-auto-tool-choice --tool-call-parser hermes \\
        --gpu-memory-utilization 0.6 --max-model-len 4096
    frun up examples/tools_agent.py      # port 8000, which is why vLLM is on 8001
    frun talk                            # in another terminal

A tool is an ordinary function. The model sees its name, its docstring and its
parameters; when it calls one, the function runs and the model answers with
what it returned. Anything it says first ("Let me check.") is spoken while the
tool runs, and talking over the wait cancels it.

Tool calling needs an LLM server that supports it (vLLM, SGLang, llama-server
with --jinja, or a hosted API); the in-process llama.cpp runtime can't.
"""
from typing import Literal

from fusion_runtime import LLM, STT, TTS, Agent, Turns, tool

# Stand-ins for your real systems: an order database, a shipping API.
ORDERS = {
    "1042": {"status": "shipped", "carrier": "BlueDart", "arriving": "Thursday"},
    "1043": {"status": "packing", "carrier": None, "arriving": "next week"},
}


@tool
async def order_status(order_id: str) -> dict:
    """Look up where an order is and when it will arrive.

    Args:
        order_id: The order number, digits only, as the caller reads it out.
    """
    order = ORDERS.get(order_id.strip().lstrip("#"))
    if order is None:
        return {"found": False, "hint": "ask the caller to read the number again"}
    return {"found": True, **order}


@tool(timeout_s=5)
def change_delivery(order_id: str, when: Literal["morning", "afternoon", "evening"]) -> str:
    """Move an order's delivery to a different time of day.

    Args:
        order_id: The order number.
        when: The part of the day the caller wants it delivered.
    """
    # An ordinary (not async) function runs on a worker thread, so a blocking
    # client library is fine here.
    if order_id not in ORDERS:
        return "no such order"
    return f"delivery for {order_id} moved to the {when}"


agent = Agent(
    name="shopkart-orders",
    prompt=(
        "You are the order line for ShopKart. Use the tools to check orders and change deliveries; "
        "never guess an order's status. Before a lookup, say a few words like 'Let me check.' "
        "Keep every answer to one short sentence."
    ),
    stt=STT("whisper-small"),
    # The model on the vLLM started above, asked for by the repo id it loaded. vLLM's own
    # default port is 8000, the same as frun up's, so on one machine it moves to 8001.
    llm=LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ", url="http://localhost:8001/v1",
            max_tokens=200, max_tool_rounds=3),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
    tools=[order_status, change_delivery],
    profile="production",  # speech on the GPU; the development profile keeps Whisper on the CPU
)
