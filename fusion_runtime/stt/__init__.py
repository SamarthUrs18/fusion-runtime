"""Speech-to-text engines."""
from fusion_runtime.stt.base import STTBase, STTResult
from fusion_runtime.stt.whisper import FasterWhisperSTT


def create_stt(config) -> STTBase:
    """Factory function to create STT instance from config."""
    from fusion_runtime.config import Provider
    
    if config.provider == Provider.FASTER_WHISPER:
        return FasterWhisperSTT(config)
    raise ValueError(f"Unknown STT provider: {config.provider}")
