"""Voice activity detection. Turn detectors live in fusion_runtime.turns."""
from fusion_runtime.vad.base import SpeechSegment, VADBase, VADResult
from fusion_runtime.vad.silero import SileroVAD, vad_stream_segments
from fusion_runtime.vad.turn import TurnState


def create_vad(config) -> VADBase:
    from fusion_runtime.config import Provider
    if config.provider == Provider.SILERO:
        return SileroVAD(config)
    raise ValueError(f"Unknown VAD provider: {config.provider}")
