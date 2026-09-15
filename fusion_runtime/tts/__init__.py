"""Text-to-speech engines."""
from fusion_runtime.tts.base import TTSBase, TTSResult
from fusion_runtime.tts.kokoro import KokoroTTS


def create_tts(config) -> TTSBase:
    """Factory function to create TTS instance from config."""
    from fusion_runtime.config import Provider
    
    if config.provider == Provider.KOKORO:
        return KokoroTTS(config)
    raise ValueError(f"Unknown TTS provider: {config.provider}")
