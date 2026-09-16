"""Text-to-speech runtime contract."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from fusion_runtime.contract.common import ModelRuntime, Request


@dataclass(kw_only=True)
class TTSRequest(Request):
    text: str  # one segment (usually a sentence); the engine does the splitting
    voice: Optional[str] = None  # None = runtime default
    speed: float = 1.0


@dataclass
class AudioChunk:
    """Mono signed 16-bit little-endian PCM at `sample_rate` Hz."""

    pcm: bytes
    sample_rate: int


class TTSRuntime(ModelRuntime):
    stage = "tts"

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        """Stream audio for one text segment.

        Chunks use `capabilities.sample_rate`; the engine resamples. On
        cancellation, stop promptly and raise Cancelled.
        """
