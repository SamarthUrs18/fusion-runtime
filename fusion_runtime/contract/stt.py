"""Speech-to-text runtime contract."""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from fusion_runtime.contract.common import AdapterError, ModelRuntime, Request

PCM16_BYTES_PER_SAMPLE = 2


@dataclass(kw_only=True)
class STTRequest(Request):
    """One utterance to transcribe: mono signed 16-bit little-endian PCM."""

    audio: bytes
    sample_rate: int = 16000
    prompt: Optional[str] = None  # context hint (names, earlier text), if the model supports one


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None  # detected or used
    confidence: Optional[float] = None
    duration_s: float = 0.0
    extra: dict = field(default_factory=dict)


STTResult = Union[Transcript, AdapterError]


class STTRuntime(ModelRuntime):
    stage = "stt"

    @abstractmethod
    async def transcribe(self, requests: Sequence[STTRequest]) -> List[STTResult]:
        """Transcribe a batch of utterances.

        Returns one result per request, in the same order. A request that
        fails or is cancelled gets an AdapterError in its slot (Cancelled for
        cancellation) instead of failing the whole batch. The engine sends at
        most `capabilities.max_batch` requests per call.
        """
