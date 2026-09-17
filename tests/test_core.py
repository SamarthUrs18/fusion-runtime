"""
Tests for fusion-runtime core components.
"""
import pytest
import asyncio
import os
from fusion_runtime.config import (
    PipelineConfig,
    STTConfig,
    LLMConfig,
    TTSConfig,
    Provider,
    DEVELOPMENT_CONFIG,
    HYBRID_CONFIG,
)
from fusion_runtime.contract import LLMRuntime, STTRuntime, TTSRuntime
from fusion_runtime.registry import create_runtime, runtime_class
from fusion_runtime.resolver import resolve_stage_config
from fusion_runtime.vad import create_vad
from fusion_runtime.engine import PipelineOrchestrator, run_single_turn


class TestConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.stt.provider == Provider.FASTER_WHISPER
        assert config.llm.provider == Provider.LLAMA_CPP
        assert config.tts.provider == Provider.KOKORO
        assert config.target_latency_ms == 500
        assert config.allow_cloud_fallback == False
    
    def test_development_config(self):
        config = DEVELOPMENT_CONFIG
        assert config.stt.device == "cpu"
        assert config.llm.n_gpu_layers == 0
    
    def test_hybrid_config(self):
        config = HYBRID_CONFIG
        assert config.allow_cloud_fallback == True
        assert config.llm.provider == Provider.OPENAI


class TestBuiltinRuntimes:
    def test_ctranslate2_is_an_stt_runtime(self):
        assert issubclass(runtime_class("stt", "ctranslate2"), STTRuntime)

    def test_llama_cpp_and_openai_http_are_llm_runtimes(self):
        assert issubclass(runtime_class("llm", "llama_cpp"), LLMRuntime)
        assert issubclass(runtime_class("llm", "openai_http"), LLMRuntime)

    def test_onnx_is_a_tts_runtime(self):
        assert issubclass(runtime_class("tts", "onnx"), TTSRuntime)

    def test_unknown_provider_raises(self):
        # Pydantic rejects unknown providers at validation time
        with pytest.raises(Exception):
            STTConfig(provider="unknown")

    def test_hybrid_llm_builds_an_http_runtime_without_loading(self, tmp_path):
        runtime = create_runtime(resolve_stage_config("llm", HYBRID_CONFIG.llm, root=tmp_path).spec)
        assert isinstance(runtime, LLMRuntime) and runtime.client is None


class TestVADFactory:
    def test_create_silero(self):
        from fusion_runtime.config import VADConfig
        config = VADConfig(provider=Provider.SILERO)
        vad = create_vad(config)
        assert vad is not None


class TestTurnDetectors:
    def test_default_is_the_silence_detector(self):
        from fusion_runtime.config import TurnDetectionConfig
        from fusion_runtime.contract import TurnDetector
        from fusion_runtime.turns import turn_detector_spec

        detector = create_runtime(turn_detector_spec(TurnDetectionConfig()))
        assert isinstance(detector, TurnDetector) and detector.refines_wait is False

    def test_a_plugin_detector_is_named_in_config(self):
        from fusion_runtime.config import TurnDetectionConfig
        from fusion_runtime.turns import turn_detector_spec

        spec = turn_detector_spec(TurnDetectionConfig(runtime="my_pkg.turns:Detector", model="hf:org/eou", options={"x": 1}))
        assert (spec.stage, spec.runtime, spec.model, dict(spec.options)) == ("turn", "my_pkg.turns:Detector", "hf:org/eou", {"x": 1})


# Integration tests (require models downloaded)
@pytest.mark.integration
class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def orchestrator(self):
        from fusion_runtime.engine import PipelineOrchestrator
        orch = PipelineOrchestrator(DEVELOPMENT_CONFIG)
        asyncio.run(orch.initialize())
        yield orch
        asyncio.run(orch.shutdown())
    
    @pytest.fixture(scope="class")
    def speech_audio(self):
        """Real speech WAV (16kHz mono int16) generated as a test fixture."""
        import wave
        path = os.path.join(os.path.dirname(__file__), "fixtures", "hello.wav")
        with wave.open(path, "rb") as wav:
            return wav.readframes(wav.getnframes())
    
    async def test_single_turn(self, orchestrator, speech_audio):
        result = await run_single_turn(orchestrator, speech_audio)
        assert isinstance(result, bytes)
        assert len(result) > 0, "Pipeline produced no audio for real speech input"
    
    async def test_streaming_pipeline(self, orchestrator, speech_audio):
        # Feed the speech audio in 100ms chunks
        chunk_ms = 100
        bytes_per_chunk = 16000 * 2 * chunk_ms // 1000  # 3200 bytes
        
        async def audio_stream():
            for i in range(0, len(speech_audio), bytes_per_chunk):
                yield speech_audio[i:i + bytes_per_chunk]
                await asyncio.sleep(chunk_ms / 1000)
        
        chunks = []
        async for chunk in orchestrator.run_pipeline(audio_stream()):
            chunks.append(chunk)
        
        assert len(chunks) > 0, "Streaming pipeline produced no audio chunks"
        assert any(len(c) > 0 for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])