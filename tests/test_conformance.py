"""Conformance kit: correct runtimes pass, and each kind of broken runtime fails the right check."""
import asyncio

import pytest

from fusion_runtime.contract import AudioChunk, Cancelled, LLMChunk, Transcript
from fusion_runtime.testing.conformance import assert_conforms, check_runtime
from fusion_runtime.testing.fakes import FakeLLMRuntime, FakeSTTRuntime, FakeTTSRuntime, fake_spec


@pytest.mark.parametrize("cls, stage, options", [
    (FakeSTTRuntime, "stt", {}),
    (FakeSTTRuntime, "stt", {"languages": ["en", "hi"], "max_batch": 1, "max_concurrency": 1}),
    (FakeLLMRuntime, "llm", {}),
    (FakeTTSRuntime, "tts", {}),
    (FakeTTSRuntime, "tts", {"languages": ["hi"], "sample_rate": 16000}),
])
async def test_well_behaved_fakes_conform(cls, stage, options):
    report = await check_runtime(cls(fake_spec(stage, **options)))
    assert_conforms(report)


def failed(report, check_name):
    outcome = report.outcome(check_name)
    assert not outcome.passed, f"expected {check_name!r} to fail:\n{report}"
    return outcome.detail


# ---- broken TTS runtimes ------------------------------------------------------------

class IgnoresCancel(FakeTTSRuntime):
    async def synthesize(self, request):
        for _ in range(30):
            await asyncio.sleep(0.02)  # never looks at request.cancel
            yield AudioChunk(pcm=bytes(960), sample_rate=24000)


class WrongSampleRate(FakeTTSRuntime):
    async def synthesize(self, request):
        async for chunk in super().synthesize(request):
            yield AudioChunk(pcm=chunk.pcm, sample_rate=16000)  # declares 24000


class OddBytes(FakeTTSRuntime):
    async def synthesize(self, request):
        yield AudioChunk(pcm=b"\x01\x02\x03", sample_rate=24000)


class PlainValueError(FakeTTSRuntime):
    async def synthesize(self, request):
        if not request.text.strip():
            raise ValueError("empty")  # should be InvalidRequest
        async for chunk in super().synthesize(request):
            yield chunk


class Hangs(FakeTTSRuntime):
    async def synthesize(self, request):
        await asyncio.Event().wait()
        yield  # pragma: no cover


class UnhealthyAfterLoad(FakeTTSRuntime):
    def health(self):
        from fusion_runtime.contract import Health
        return Health("down", "GPU lost")


async def test_tts_ignoring_cancel_fails():
    report = await check_runtime(IgnoresCancel(fake_spec("tts")), cancel_timeout_s=0.3)
    assert "after cancel" in failed(report, "stops promptly when cancelled mid-stream")


async def test_tts_wrong_sample_rate_fails():
    report = await check_runtime(WrongSampleRate(fake_spec("tts")))
    assert "!= declared 24000" in failed(report, "produces 16-bit PCM at the declared sample rate")


async def test_tts_odd_byte_count_fails():
    report = await check_runtime(OddBytes(fake_spec("tts")))
    assert "not 16-bit PCM" in failed(report, "produces 16-bit PCM at the declared sample rate")


async def test_tts_wrong_error_type_fails():
    report = await check_runtime(PlainValueError(fake_spec("tts")))
    assert "use the contract's error types" in failed(report, "rejects empty text with InvalidRequest")


async def test_hanging_runtime_times_out_instead_of_hanging_tests():
    report = await check_runtime(Hangs(fake_spec("tts")), timeout_s=0.2)
    assert "timed out" in failed(report, "produces 16-bit PCM at the declared sample rate")


async def test_unhealthy_runtime_fails():
    report = await check_runtime(UnhealthyAfterLoad(fake_spec("tts")))
    assert "GPU lost" in failed(report, "declares capabilities and health")


async def test_failed_load_stops_the_run():
    class LoadFails(FakeTTSRuntime):
        async def load(self):
            raise RuntimeError("weights missing")

    report = await check_runtime(LoadFails(fake_spec("tts")))
    assert [o.name for o in report.outcomes] == ["loads"]
    assert "weights missing" in failed(report, "loads")


async def test_assert_conforms_lists_failures():
    report = await check_runtime(OddBytes(fake_spec("tts")))
    with pytest.raises(AssertionError, match="✗ produces 16-bit PCM"):
        assert_conforms(report)


# ---- broken LLM runtimes ------------------------------------------------------------

class NoFinishReason(FakeLLMRuntime):
    async def generate(self, request):
        async for chunk in super().generate(request):
            yield LLMChunk(text=chunk.text)


class FinishReasonTooEarly(FakeLLMRuntime):
    async def generate(self, request):
        async for chunk in super().generate(request):
            yield LLMChunk(text=chunk.text, finish_reason="stop")


async def test_llm_without_finish_reason_fails():
    report = await check_runtime(NoFinishReason(fake_spec("llm")))
    assert "no finish_reason" in failed(report, "streams chunks and marks the last with finish_reason")


async def test_llm_finish_reason_before_last_chunk_fails():
    report = await check_runtime(FinishReasonTooEarly(fake_spec("llm")))
    assert "before the last chunk" in failed(report, "streams chunks and marks the last with finish_reason")


# ---- broken STT runtimes ------------------------------------------------------------

class DropsResults(FakeSTTRuntime):
    async def transcribe(self, requests):
        return (await super().transcribe(requests))[:1]


class RaisesForWholeBatch(FakeSTTRuntime):
    async def transcribe(self, requests):
        for r in requests:
            if r.cancel.cancelled:
                raise Cancelled()  # should be a Cancelled in that request's slot
        return await super().transcribe(requests)


class IgnoresLanguage(FakeSTTRuntime):
    async def _one(self, request):
        return Transcript(text="ok")  # claims languages but never checks them


async def test_stt_dropping_results_fails():
    report = await check_runtime(DropsResults(fake_spec("stt")))
    assert "3 requests gave 1 results" in failed(report, "returns one result per request, in a list")


async def test_stt_failing_whole_batch_for_one_cancel_fails():
    report = await check_runtime(RaisesForWholeBatch(fake_spec("stt")))
    failed(report, "reports a cancelled request as Cancelled in its own slot")


async def test_stt_ignoring_declared_languages_fails():
    report = await check_runtime(IgnoresLanguage(fake_spec("stt", languages=["en"])))
    failed(report, "reports an unsupported language as InvalidRequest")
    failed(report, "reports invalid audio as InvalidRequest in its own slot")
