"""The runtime contract: cancellation, capabilities, requests and specs."""
import asyncio
import threading
import time

import pytest

from fusion_runtime.contract import (
    Cancelled,
    CancelToken,
    Capabilities,
    ModelSpec,
    RateLimited,
    TTSRequest,
)
from fusion_runtime.testing.fakes import FakeTTSRuntime, fake_spec


# ---- CancelToken ---------------------------------------------------------------------

def test_cancel_sets_flag_and_reason_once():
    token = CancelToken()
    assert not token.cancelled
    token.cancel("barge-in")
    token.cancel("second reason is ignored")
    assert token.cancelled and token.reason == "barge-in"


def test_callbacks_run_once_and_late_callbacks_run_immediately():
    token, calls = CancelToken(), []
    token.add_callback(lambda: calls.append("early"))
    token.cancel()
    token.cancel()
    token.add_callback(lambda: calls.append("late"))
    assert calls == ["early", "late"]


def test_raise_if_cancelled():
    token = CancelToken()
    token.raise_if_cancelled()
    token.cancel("stop")
    with pytest.raises(Cancelled, match="stop"):
        token.raise_if_cancelled()


async def test_wait_wakes_when_cancelled_from_another_thread():
    token = CancelToken()
    threading.Timer(0.05, token.cancel).start()  # like a decode thread or another session cancelling
    start = time.monotonic()
    assert await token.wait(timeout=2) is True
    assert time.monotonic() - start < 1


async def test_wait_times_out_without_leaking_callbacks():
    token = CancelToken()
    assert await token.wait(timeout=0.01) is False
    assert token._callbacks == []


# ---- Capabilities and requests ---------------------------------------------------------

def test_capabilities_validate_limits():
    with pytest.raises(ValueError):
        Capabilities(max_batch=0)
    with pytest.raises(ValueError):
        Capabilities(max_concurrency=0)
    with pytest.raises(ValueError):
        Capabilities(sample_rate=0)


def test_language_matching():
    caps = Capabilities(languages=("en-us", "hi", "pt-br"))
    assert caps.supports_language("hi")
    assert caps.supports_language("en")  # base language of en-us
    assert caps.supports_language("PT-BR")
    assert not caps.supports_language("ja")
    assert caps.supports_language(None)
    assert Capabilities().supports_language("anything")  # no list = any


def test_requests_get_unique_ids_and_own_cancel_tokens():
    a, b = TTSRequest(text="hi"), TTSRequest(text="hi")
    assert a.id != b.id
    a.cancel.cancel()
    assert not b.cancel.cancelled


def test_remaining_time():
    assert TTSRequest(text="hi").remaining_s() is None
    r = TTSRequest(text="hi", deadline=time.monotonic() + 5)
    assert 4 < r.remaining_s() <= 5


def test_retryable_errors():
    err = RateLimited(retry_after_s=2)
    assert err.retryable and err.retry_after_s == 2
    assert not Cancelled().retryable


def test_runtime_rejects_spec_for_another_stage():
    with pytest.raises(ValueError, match="tts runtime, got a llm"):
        FakeTTSRuntime(ModelSpec(stage="llm", runtime="x", model="y"))


def test_fake_llm_decodes_only_when_pulled():
    # The contract says LLM runtimes are pull-driven; the fake is the reference.
    from fusion_runtime.contract import LLMRequest, Message
    from fusion_runtime.testing.fakes import FakeLLMRuntime

    async def run():
        llm = FakeLLMRuntime(fake_spec("llm", step_s=0.001))
        stream = llm.generate(LLMRequest(messages=[Message("user", "hi")]))
        await stream.__anext__()
        await asyncio.sleep(0.05)  # caller busy
        decoded = llm.decoded
        await stream.aclose()
        return decoded

    assert asyncio.run(run()) == 1
