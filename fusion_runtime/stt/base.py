"""Speech-to-text interface shared by every STT engine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import asyncio


@dataclass
class STTResult:
    """A streaming transcription result.

    `text` is **cumulative for the current turn**: each result restates the
    whole turn transcribed so far, superseding the previous one, rather than
    carrying only newly-recognized words. Consumers must therefore *replace*
    their running transcript with it, not append — appending overlapping
    re-transcriptions is what produced garbled, duplicated transcripts.
    A turn's accumulation is reset via `transcribe_stream`'s `reset_signal`.
    """
    text: str
    is_final: bool
    confidence: float
    latency_ms: float
    language: Optional[str] = None


class STTBase(ABC):
    """Base class for all STT providers."""
    
    def __init__(self, config):
        self.config = config
        self._warm = False
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        budget_ms: Optional[int] = None,
        reset_signal: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[STTResult]:
        """Stream transcription with partial results.

        `reset_signal`, when provided, is set by the caller once a turn has
        ended — implementations using a rolling/sliding audio window should
        clear it on the next chunk so old, already-transcribed speech isn't
        re-emitted into the next turn.
        """
        pass
    
    @abstractmethod
    async def transcribe_file(self, audio_path: str) -> STTResult:
        """Transcribe complete audio file."""
        pass
    
    async def warmup(self):
        """Pre-load models, warm up GPU."""
        if not self._warm:
            await self._warmup_impl()
            self._warm = True
    
    @abstractmethod
    async def _warmup_impl(self):
        pass
