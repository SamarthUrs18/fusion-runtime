"""Built-in runtimes: error handling without models, and conformance with real models when downloaded."""
import wave
from pathlib import Path

import pytest

from fusion_runtime.config import DEVELOPMENT_CONFIG, PipelineConfig
from fusion_runtime.contract import (
    ModelNotFound, ModelSpec, UnsupportedModel,
)
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.engine.orchestrator import _resample_pcm16
from fusion_runtime.registry import create_runtime
from fusion_runtime.resolver import ResolvedModel, resolve_stage_config
from fusion_runtime.runtimes.onnx.tts import OnnxTTS
from fusion_runtime.telemetry import ListSink, telemetry
from fusion_runtime.testing.conformance import assert_conforms, check_runtime
from fusion_runtime.testing.fakes import fake_spec

FIXTURE = Path(__file__).parent / "fixtures" / "hello.wav"


def test_onnx_rejects_an_unknown_family():
    with pytest.raises(UnsupportedModel, match="known families: kokoro"):
        OnnxTTS(ModelSpec(stage="tts", runtime="onnx", model="x.onnx", family="nope"))


async def test_kokoro_missing_files_say_how_to_get_them(tmp_path):
    runtime = OnnxTTS(ModelSpec(stage="tts", runtime="onnx", model=str(tmp_path / "model.onnx"), family="kokoro"))
    with pytest.raises(ModelNotFound, match="frun models pull"):
        await runtime.load()


async def test_llama_cpp_missing_file(tmp_path):
    runtime = create_runtime(ModelSpec(stage="llm", runtime="llama_cpp", model=str(tmp_path / "none.gguf")))
    with pytest.raises(ModelNotFound):
        await runtime.load()


def test_resampling_changes_rate_and_keeps_duration():
    pcm = b"\x00\x10" * 24000
    out = _resample_pcm16(pcm, 24000, 16000)
    assert len(out) == 2 * 16000
    assert _resample_pcm16(pcm, 24000, 24000) is pcm


async def test_initialize_loads_resolved_runtimes_and_sizes_queues(monkeypatch):
    specs = {
        "stt": fake_spec("stt", max_concurrency=3),
        "llm": fake_spec("llm", max_concurrency=2),
        "tts": fake_spec("tts"),
    }
    targets = {
        "stt": "fusion_runtime.testing.fakes:FakeSTTRuntime",
        "llm": "fusion_runtime.testing.fakes:FakeLLMRuntime",
        "tts": "fusion_runtime.testing.fakes:FakeTTSRuntime",
    }

    def fake_resolve(stage, stage_config, **_):
        spec = specs[stage]
        spec = ModelSpec(stage=stage, runtime=targets[stage], model="fake", options=spec.options)
        return ResolvedModel(spec, "plugin", None)

    monkeypatch.setattr("fusion_runtime.resolver.resolve_stage_config", fake_resolve)
    orch = PipelineOrchestrator(PipelineConfig())

    async def no_vad():
        return None

    monkeypatch.setattr(orch, "_load_vad_frame_model", no_vad)
    sink = ListSink()
    telemetry.add_sink(sink)
    try:
        await orch.initialize()
    finally:
        telemetry.remove_sink(sink)
    assert orch.ready and orch.llm.loaded and orch.stt.loaded and orch.tts.loaded
    assert orch.scheduler("stt").max_concurrency == 3 and orch.scheduler("llm").max_concurrency == 2
    loaded = {e.stage: e.attrs for e in sink.named("model.loaded")}
    assert loaded["llm"]["runtime"].endswith("FakeLLMRuntime")
    await orch.shutdown()
    assert orch.llm.closed and not orch.ready


async def test_a_model_that_doesnt_resolve_fails_startup_with_a_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path))
    orch = PipelineOrchestrator(PipelineConfig())
    sink = ListSink()
    telemetry.add_sink(sink)
    try:
        with pytest.raises(ModelNotFound):
            await orch._load_runtime("llm")
    finally:
        telemetry.remove_sink(sink)
    failure = sink.named("model.load_failed")[0]
    assert failure.error.code == "model_not_found" and "frun models pull" in failure.error.message


# ---- real models (skipped when not downloaded) ----------------------------------------------

def _resolved_or_skip(stage):
    try:
        return resolve_stage_config(stage, getattr(DEVELOPMENT_CONFIG, stage))
    except ModelNotFound as e:
        pytest.skip(f"{stage} model not downloaded: {e}")


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["stt", "llm", "tts"])
async def test_real_runtime_conforms(stage):
    resolved = _resolved_or_skip(stage)
    runtime = create_runtime(resolved.spec)
    kwargs = {}
    if stage == "stt":
        with wave.open(str(FIXTURE), "rb") as wav:
            kwargs["stt_audio"] = wav.readframes(wav.getnframes())
    assert_conforms(await check_runtime(runtime, **kwargs))
