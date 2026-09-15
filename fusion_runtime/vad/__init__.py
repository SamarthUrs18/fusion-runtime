"""Voice activity detection and turn detection."""
from fusion_runtime.vad.base import SpeechSegment, VADBase, VADResult
from fusion_runtime.vad.silero import SileroVAD, vad_stream_segments
from fusion_runtime.vad.turn import PunctuationTurnDetector, TurnDetectorBase, TurnState


def create_vad(config) -> VADBase:
    from fusion_runtime.config import Provider
    if config.provider == Provider.SILERO:
        return SileroVAD(config)
    raise ValueError(f"Unknown VAD provider: {config.provider}")


def create_turn_detector(config) -> TurnDetectorBase:
    from fusion_runtime.config import Provider
    if config.provider == Provider.PUNCTUATION:
        return PunctuationTurnDetector(config)
    raise ValueError(f"Unknown turn detector: {config.provider}")
