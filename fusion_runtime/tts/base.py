"""Text-to-speech interface shared by every TTS engine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List


@dataclass
class TTSResult:
    audio: bytes  # Raw PCM or encoded
    is_final: bool
    sample_rate: int
    latency_ms: float
    format: str = "pcm"  # pcm, mp3, wav


class TTSBase(ABC):
    """Base class for all TTS providers."""
    
    def __init__(self, config):
        self.config = config
        self._warm = False
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        budget_ms: Optional[int] = None
    ) -> AsyncIterator[TTSResult]:
        """Stream synthesis from text tokens."""
        pass
    
    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize complete text."""
        pass
    
    async def warmup(self):
        if not self._warm:
            await self._warmup_impl()
            self._warm = True
    
    @abstractmethod
    async def _warmup_impl(self):
        pass
