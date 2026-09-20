"""Voice-activity-detection interface."""
from abc import ABC, abstractmethod


class VADBase(ABC):
    """Configuration for a voice-activity detector.

    Detection itself happens frame by frame in the engine, which holds one model
    copy per stream. An implementation supplies the settings for that and a way
    to drop per-stream state between turns.
    """

    def __init__(self, config):
        self.config = config
        self.sample_rate = getattr(config, 'sample_rate', 16000)

    @abstractmethod
    async def reset(self):
        """Forget anything carried over from the previous stream."""
