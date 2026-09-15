"""Voice-activity-detection interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class VADResult:
    is_speech: bool
    confidence: float
    audio: bytes  # Pass-through audio
    start_sample: int = 0
    end_sample: int = 0


@dataclass
class SpeechSegment:
    """A detected speech segment with audio and timestamps."""
    audio: bytes
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    confidence: float


class VADBase(ABC):
    """Base class for Voice Activity Detection."""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = getattr(config, 'sample_rate', 16000)
    
    @abstractmethod
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[SpeechSegment]:
        """Process audio stream and yield speech segments."""
        pass
    
    @abstractmethod
    async def reset(self):
        pass
