"""State shared between voice activity detection and turn detection."""
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class TurnState:
    """VAD-derived signal shared between the audio pipeline and turn
    detection: how long (ms) it's been since Silero last saw real speech.

    Updated frame-by-frame by `PipelineOrchestrator._apply_vad`; read by
    the turn detector so it can tell a genuine pause apart from the STT's
    own rolling-window transcript merely ending in punctuation (Whisper
    does that constantly on short windows, even mid-sentence).
    `vad_active` is False until real VAD frames have been scored (e.g. the
    Silero model failed to load) — callers should skip silence-gating in
    that case rather than silently never reaching the threshold.
    """
    silence_ms: float = 0.0
    vad_active: bool = False
    # This turn's speech so far (16-bit PCM, 16 kHz), kept only when the turn
    # detector uses audio; trimmed to the most recent seconds.
    speech_audio: bytearray = field(default_factory=bytearray)
    # Set by the STT stage: transcribe the words since the last partial (returns a
    # PartialTranscript or None), and start the next turn's transcript right away.
    finalize_transcript: Optional[Callable[[], Awaitable[Any]]] = None
    start_next_turn: Optional[Callable[[], None]] = None
