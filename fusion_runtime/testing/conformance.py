"""Conformance kit: behaviour every runtime must have, whoever wrote it.

Use it in a runtime's own tests:

    from fusion_runtime.testing.conformance import assert_conforms, check_runtime

    async def test_my_runtime_conforms():
        runtime = MyTTSRuntime(spec)
        assert_conforms(await check_runtime(runtime, tts_voice="alba"))

Each check has a timeout, so a runtime that hangs fails a check instead of
hanging the test run. The kit loads the runtime first and closes it at the end.
Real runtimes can pass sample inputs (real speech, a valid voice) through
check_runtime's keyword arguments.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from fusion_runtime.contract import (
    AdapterError,
    AudioChunk,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    LLMChunk,
    LLMRequest,
    LLMRuntime,
    Message,
    ModelRuntime,
    STTRequest,
    STTRuntime,
    Transcript,
    TTSRequest,
    TTSRuntime,
)

UNSUPPORTED_LANGUAGE = "zz"  # a code no runtime will claim


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceReport:
    runtime: str
    outcomes: List[CheckOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    @property
    def failures(self) -> List[CheckOutcome]:
        return [o for o in self.outcomes if not o.passed]

    def outcome(self, name: str) -> CheckOutcome:
        return next(o for o in self.outcomes if o.name == name)

    def __str__(self) -> str:
        lines = [f"Conformance for {self.runtime}: "
                 f"{len(self.outcomes) - len(self.failures)}/{len(self.outcomes)} checks passed"]
        for o in self.outcomes:
            lines.append(f"  {'✓' if o.passed else '✗'} {o.name}" + (f": {o.detail}" if o.detail else ""))
        return "\n".join(lines)


class CheckFailed(AssertionError):
    pass


def assert_conforms(report: ConformanceReport) -> None:
    if not report.passed:
        raise AssertionError(str(report))


async def check_runtime(
    runtime: ModelRuntime,
    *,
    timeout_s: float = 10.0,
    load_timeout_s: float = 300.0,
    cancel_timeout_s: float = 1.0,
    stt_audio: Optional[bytes] = None,
    llm_prompt: str = "Say hello.",
    tts_text: str = "Hello there. This is a short test sentence.",
    tts_voice: Optional[str] = None,
) -> ConformanceReport:
    report = ConformanceReport(runtime=type(runtime).__name__)

    async def run(name: str, check: Callable[[], Awaitable[Optional[str]]], timeout: float = timeout_s) -> bool:
        try:
            detail = await asyncio.wait_for(check(), timeout)
            report.outcomes.append(CheckOutcome(name, True, detail or ""))
            return True
        except asyncio.TimeoutError:
            report.outcomes.append(CheckOutcome(name, False, f"timed out after {timeout:g}s"))
        except CheckFailed as e:
            report.outcomes.append(CheckOutcome(name, False, str(e)))
        except Exception as e:
            report.outcomes.append(CheckOutcome(name, False, f"unexpected {type(e).__name__}: {e}"))
        return False

    async def load() -> None:
        await runtime.load()

    if not await run("loads", load, load_timeout_s):
        return report  # nothing else can be checked

    await run("declares capabilities and health", lambda: _check_capabilities(runtime))

    if isinstance(runtime, TTSRuntime):
        await _tts_checks(runtime, run, cancel_timeout_s, tts_text, tts_voice)
    elif isinstance(runtime, LLMRuntime):
        await _llm_checks(runtime, run, cancel_timeout_s, llm_prompt)
    elif isinstance(runtime, STTRuntime):
        await _stt_checks(runtime, run, stt_audio or bytes(16000 * 2))
    else:
        report.outcomes.append(CheckOutcome("is a known stage", False, "not an STT, LLM or TTS runtime"))

    async def close_twice() -> None:
        await runtime.close()
        await runtime.close()

    await run("closes, and closing twice is safe", close_twice)
    return report


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


async def _check_capabilities(runtime: ModelRuntime) -> str:
    caps = runtime.capabilities
    _require(isinstance(caps, Capabilities), f"capabilities is {type(caps).__name__}, not Capabilities")
    if runtime.stage in ("stt", "tts"):
        _require(caps.sample_rate is not None, "audio runtimes must declare sample_rate")
    health = runtime.health()
    _require(isinstance(health, Health), f"health() returned {type(health).__name__}, not Health")
    _require(health.ok, f"health is {health.status} right after load: {health.detail}")
    return f"max_batch={caps.max_batch} max_concurrency={caps.max_concurrency}"


async def _expect_raises(awaitable_factory, error: type, message: str) -> None:
    try:
        await awaitable_factory()
    except error:
        return
    except AdapterError as e:
        raise CheckFailed(f"{message}: raised {type(e).__name__} instead of {error.__name__}")
    except Exception as e:
        raise CheckFailed(f"{message}: raised {type(e).__name__} (use the contract's error types)")
    raise CheckFailed(f"{message}: nothing was raised")


async def _drain(stream) -> list:
    return [item async for item in stream]


# ---- streaming stages (TTS, LLM) ---------------------------------------------------

async def _cancel_mid_stream(make_stream, request, cancel_timeout_s: float) -> str:
    stream = make_stream(request)
    try:
        await stream.__anext__()
    except StopAsyncIteration:
        return "finished in one chunk; mid-stream cancellation not exercised"
    request.cancel.cancel("conformance check")
    extra = 0

    async def finish():
        nonlocal extra
        async for _ in stream:
            extra += 1

    try:
        await asyncio.wait_for(finish(), cancel_timeout_s)
    except Cancelled:
        pass
    except asyncio.TimeoutError:
        raise CheckFailed(f"kept streaming for over {cancel_timeout_s:g}s after cancel")
    _require(extra <= 1, f"produced {extra} more chunks after cancel (at most 1 allowed)")
    return ""


async def _abandon_then_reuse(make_stream, make_request) -> None:
    stream = make_stream(make_request())
    try:
        await stream.__anext__()
    except StopAsyncIteration:
        pass
    await stream.aclose()
    items = await _drain(make_stream(make_request()))
    _require(len(items) > 0, "no output from a request made after abandoning a stream")


async def _concurrent(make_stream, make_request, count: int) -> str:
    results = await asyncio.gather(*(_drain(make_stream(make_request())) for _ in range(count)))
    _require(all(len(items) > 0 for items in results), "a concurrent request produced no output")
    return f"{count} at once"


async def _tts_checks(runtime: TTSRuntime, run, cancel_timeout_s: float, text: str, voice: Optional[str]) -> None:
    caps = runtime.capabilities

    def request(**overrides) -> TTSRequest:
        return TTSRequest(**{"text": text, "voice": voice, **overrides})

    async def produces_audio():
        chunks = await _drain(runtime.synthesize(request()))
        _require(len(chunks) > 0, "no audio chunks")
        _require(all(isinstance(c, AudioChunk) for c in chunks), "yielded something other than AudioChunk")
        _require(all(c.pcm and len(c.pcm) % 2 == 0 for c in chunks), "empty chunk or odd byte count (not 16-bit PCM)")
        rates = {c.sample_rate for c in chunks}
        _require(rates == {caps.sample_rate}, f"chunk sample rates {sorted(rates)} != declared {caps.sample_rate}")
        seconds = sum(len(c.pcm) for c in chunks) / 2 / caps.sample_rate
        return f"{len(chunks)} chunks, {seconds:.2f}s"

    async def rejects_empty_text():
        await _expect_raises(lambda: _drain(runtime.synthesize(request(text="   "))), InvalidRequest, "empty text")

    async def cancel_before_start():
        r = request()
        r.cancel.cancel("conformance check")
        await _expect_raises(lambda: _drain(runtime.synthesize(r)), Cancelled, "already-cancelled request")

    await run("produces 16-bit PCM at the declared sample rate", produces_audio)
    await run("rejects empty text with InvalidRequest", rejects_empty_text)
    await run("raises Cancelled for an already-cancelled request", cancel_before_start)
    await run("stops promptly when cancelled mid-stream",
              lambda: _cancel_mid_stream(runtime.synthesize, request(), cancel_timeout_s))
    await run("works after a stream is abandoned", lambda: _abandon_then_reuse(runtime.synthesize, request))
    await run("serves concurrent requests up to max_concurrency",
              lambda: _concurrent(runtime.synthesize, request, max(1, min(caps.max_concurrency, 3))))
    if caps.languages is not None:
        await run("rejects an unsupported language with InvalidRequest",
                  lambda: _expect_raises(lambda: _drain(runtime.synthesize(request(language=UNSUPPORTED_LANGUAGE))),
                                         InvalidRequest, "unsupported language"))


async def _llm_checks(runtime: LLMRuntime, run, cancel_timeout_s: float, prompt: str) -> None:
    caps = runtime.capabilities

    def request(**overrides) -> LLMRequest:
        return LLMRequest(**{"messages": [Message("user", prompt)], "max_tokens": 32, **overrides})

    async def streams_then_finishes():
        chunks = await _drain(runtime.generate(request()))
        _require(len(chunks) > 0, "no chunks")
        _require(all(isinstance(c, LLMChunk) for c in chunks), "yielded something other than LLMChunk")
        _require(chunks[-1].finish_reason is not None, "last chunk has no finish_reason")
        _require(all(c.finish_reason is None for c in chunks[:-1]), "finish_reason set before the last chunk")
        _require(any(c.text or c.tool_calls for c in chunks), "no text or tool calls in the reply")
        return f"{len(chunks)} chunks, finish_reason={chunks[-1].finish_reason}"

    async def rejects_empty_messages():
        await _expect_raises(lambda: _drain(runtime.generate(request(messages=[]))), InvalidRequest, "empty messages")

    async def cancel_before_start():
        r = request()
        r.cancel.cancel("conformance check")
        await _expect_raises(lambda: _drain(runtime.generate(r)), Cancelled, "already-cancelled request")

    await run("streams chunks and marks the last with finish_reason", streams_then_finishes)
    await run("rejects empty messages with InvalidRequest", rejects_empty_messages)
    await run("raises Cancelled for an already-cancelled request", cancel_before_start)
    await run("stops promptly when cancelled mid-stream",
              lambda: _cancel_mid_stream(runtime.generate, request(), cancel_timeout_s))
    await run("works after a stream is abandoned", lambda: _abandon_then_reuse(runtime.generate, request))
    await run("serves concurrent requests up to max_concurrency",
              lambda: _concurrent(runtime.generate, request, max(1, min(caps.max_concurrency, 3))))


async def _stt_checks(runtime: STTRuntime, run, audio: bytes) -> None:
    caps = runtime.capabilities

    def request(**overrides) -> STTRequest:
        return STTRequest(**{"audio": audio, "sample_rate": 16000, **overrides})

    async def one_result_per_request():
        count = max(1, min(caps.max_batch, 3))
        results = await runtime.transcribe([request() for _ in range(count)])
        _require(isinstance(results, list), f"returned {type(results).__name__}, not a list")
        _require(len(results) == count, f"{count} requests gave {len(results)} results")
        _require(all(isinstance(r, Transcript) for r in results),
                 "valid audio should give Transcript results, got " + ", ".join(type(r).__name__ for r in results))
        return f"batch of {count}"

    async def empty_batch():
        results = await runtime.transcribe([])
        _require(results == [], f"empty batch returned {results!r}")

    async def invalid_audio_in_its_slot():
        bad = request(audio=b"\x00")  # odd byte count: not 16-bit PCM
        results = await runtime.transcribe([bad, request()])
        _require(len(results) == 2, f"2 requests gave {len(results)} results")
        _require(isinstance(results[0], InvalidRequest), f"invalid audio gave {type(results[0]).__name__}")
        _require(isinstance(results[1], Transcript), "a bad request spoiled the rest of the batch")

    async def cancelled_in_its_slot():
        cancelled = request()
        cancelled.cancel.cancel("conformance check")
        results = await runtime.transcribe([cancelled, request()])
        _require(len(results) == 2, f"2 requests gave {len(results)} results")
        _require(isinstance(results[0], Cancelled), f"cancelled request gave {type(results[0]).__name__}")
        _require(isinstance(results[1], Transcript), "a cancelled request spoiled the rest of the batch")

    async def concurrent_batches():
        count = max(1, min(caps.max_concurrency, 3))
        outcomes = await asyncio.gather(*(runtime.transcribe([request()]) for _ in range(count)))
        _require(all(len(o) == 1 and isinstance(o[0], Transcript) for o in outcomes), "a concurrent call failed")
        return f"{count} at once"

    await run("returns one result per request, in a list", one_result_per_request)
    await run("returns [] for an empty batch", empty_batch)
    await run("reports invalid audio as InvalidRequest in its own slot", invalid_audio_in_its_slot)
    await run("reports a cancelled request as Cancelled in its own slot", cancelled_in_its_slot)
    await run("serves concurrent calls up to max_concurrency", concurrent_batches)
    if caps.languages is not None:
        async def unsupported_language():
            results = await runtime.transcribe([request(language=UNSUPPORTED_LANGUAGE)])
            _require(len(results) == 1 and isinstance(results[0], InvalidRequest),
                     f"unsupported language gave {type(results[0]).__name__ if results else 'nothing'}")
        await run("reports an unsupported language as InvalidRequest", unsupported_language)
