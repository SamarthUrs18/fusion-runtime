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
from fusion_runtime.stt import create_stt, STTBase
from fusion_runtime.llm import create_llm, LLMBase
from fusion_runtime.tts import create_tts, TTSBase
from fusion_runtime.vad import create_vad, create_turn_detector
from fusion_runtime.orchestrator import PipelineOrchestrator, run_single_turn


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


class TestSTTFactory:
    def test_create_faster_whisper(self):
        config = STTConfig(provider=Provider.FASTER_WHISPER)
        stt = create_stt(config)
        assert isinstance(stt, STTBase)
    
    def test_create_deepgram(self):
        config = STTConfig(provider=Provider.DEEPGRAM, api_key="test")
        stt = create_stt(config)
        assert isinstance(stt, STTBase)
    
    def test_unknown_provider_raises(self):
        # Pydantic rejects unknown providers at validation time
        with pytest.raises(Exception):
            STTConfig(provider="unknown")


class TestLLMFactory:
    def test_create_llama_cpp(self):
        config = LLMConfig(provider=Provider.LLAMA_CPP)
        llm = create_llm(config)
        assert isinstance(llm, LLMBase)
    
    def test_create_openai(self):
        config = LLMConfig(provider=Provider.OPENAI, api_key="test")
        llm = create_llm(config)
        assert isinstance(llm, LLMBase)


class TestTTSFactory:
    def test_create_kokoro(self):
        config = TTSConfig(provider=Provider.KOKORO)
        tts = create_tts(config)
        assert isinstance(tts, TTSBase)
    
    def test_create_elevenlabs(self):
        config = TTSConfig(provider=Provider.ELEVENLABS, api_key="test")
        tts = create_tts(config)
        assert isinstance(tts, TTSBase)


class TestVADFactory:
    def test_create_silero(self):
        from fusion_runtime.config import VADConfig
        config = VADConfig(provider=Provider.SILERO)
        vad = create_vad(config)
        assert vad is not None


class TestTurnDetectorFactory:
    def test_create_punctuation(self):
        from fusion_runtime.config import TurnDetectionConfig
        config = TurnDetectionConfig(provider=Provider.PUNCTUATION)
        detector = create_turn_detector(config)
        assert detector is not None
    
    def test_create_fire_red_eot(self):
        from fusion_runtime.config import TurnDetectionConfig
        config = TurnDetectionConfig(provider=Provider.FIRE_RED_EOT)
        detector = create_turn_detector(config)
        assert detector is not None


# Integration tests (require models downloaded)
@pytest.mark.integration
class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def orchestrator(self):
        from fusion_runtime.orchestrator import PipelineOrchestrator
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