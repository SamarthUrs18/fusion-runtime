"""LlamaCppLLM streaming: decoding must never block the event loop.

Uses a fake llama.cpp whose decode steps block with time.sleep, like the real
C calls do. Before decoding moved to a worker thread, the event loop got no
turn at all for the whole reply (measured: 1.8 s frozen on a 0.5B model), so
audio input and barge-in detection stalled until TTS paused. The thread must
also decode only on request, or it competes with TTS for the CPU.
"""
import asyncio
import time

import pytest

from fusion_runtime.config import LLMConfig, Provider
from fusion_runtime.llm import ChatMessage, LlamaCppLLM

MESSAGES = [ChatMessage(role="user", content="hi")]


class FakeLlama:
    def __init__(self, tokens=("Hello", ",", " there", "!"), step_s=0.02, fail_at=None):
        self.tokens, self.step_s, self.fail_at = list(tokens), step_s, fail_at
        self.decoded = 0
        self.active = 0
        self.max_active = 0

    def __call__(self, prompt, max_tokens, temperature, top_p, stream):
        if not stream:
            return {"choices": [{"text": "hi", "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

        def gen():
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                for i, token in enumerate(self.tokens):
                    if i == self.fail_at:
                        raise RuntimeError("decode failed")
                    time.sleep(self.step_s)  # a blocking decode step
                    self.decoded += 1
                    yield {"choices": [{"text": token}]}
            finally:
                self.active -= 1

        return gen()


def make_llm(fake: FakeLlama) -> LlamaCppLLM:
    llm = LlamaCppLLM(LLMConfig(provider=Provider.LLAMA_CPP))
    llm.llm = fake
    llm._warm = True
    return llm


async def collect(llm):
    return [r async for r in llm.generate_stream(MESSAGES)]


async def test_streams_tokens_then_final_result():
    results = await collect(make_llm(FakeLlama()))
    assert "".join(r.text for r in results) == "Hello, there!"
    assert [r.is_final for r in results] == [False, False, False, False, True]
    assert results[-1].tokens_used == 4 and results[-1].finish_reason == "stop"


async def test_event_loop_keeps_running_while_decoding():
    llm = make_llm(FakeLlama(tokens=["x"] * 10, step_s=0.05))  # 0.5 s of blocking decode
    gaps, running = [], True

    async def heartbeat():
        last = time.perf_counter()
        while running:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await collect(llm)
    running = False
    await beat
    assert len(gaps) > 20, "the loop barely ran while tokens were decoding"
    assert max(gaps) < 0.1, f"event loop blocked for {max(gaps) * 1000:.0f} ms"


async def test_decodes_only_while_the_caller_is_waiting():
    # While TTS synthesizes a sentence the pipeline stops pulling tokens. Decoding
    # ahead during that time, even one token, made CPU synthesis much slower.
    fake = FakeLlama(tokens=["x"] * 20, step_s=0.005)
    stream = make_llm(fake).generate_stream(MESSAGES)
    await stream.__anext__()
    await stream.__anext__()
    await asyncio.sleep(0.2)  # caller busy, not asking for more
    assert fake.decoded == 2, f"decoded {fake.decoded} tokens nobody asked for yet"
    await stream.aclose()


async def test_stopping_early_stops_decoding():
    fake = FakeLlama(tokens=["x"] * 100, step_s=0.01)
    llm = make_llm(fake)
    async for result in llm.generate_stream(MESSAGES):
        if result.tokens_used == 2:
            break  # what the orchestrator does on barge-in
    await asyncio.sleep(0.2)
    assert fake.decoded < 10, f"kept decoding {fake.decoded} tokens nobody will hear"
    assert fake.active == 0


async def test_concurrent_callers_take_turns_on_one_context():
    fake = FakeLlama(tokens=["a", "b", "c"], step_s=0.02)
    llm = make_llm(fake)
    first, second = await asyncio.gather(collect(llm), collect(llm))
    assert fake.max_active == 1, "two replies decoded on one llama.cpp context at once"
    assert "".join(r.text for r in first) == "abc" == "".join(r.text for r in second)


async def test_decode_errors_reach_the_caller():
    llm = make_llm(FakeLlama(fail_at=2))
    with pytest.raises(RuntimeError, match="decode failed"):
        await collect(llm)


async def test_a_new_reply_works_after_an_abandoned_one():
    fake = FakeLlama(tokens=["x"] * 50, step_s=0.01)
    llm = make_llm(fake)
    async for _ in llm.generate_stream(MESSAGES):
        break
    fake.tokens = ["ok"]
    results = await asyncio.wait_for(collect(llm), timeout=2)
    assert "".join(r.text for r in results) == "ok"
