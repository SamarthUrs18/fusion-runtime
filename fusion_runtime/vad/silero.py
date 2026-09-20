"""Silero voice-activity detection.

The model is not loaded here. Silero carries context from one frame into the
next, so two audio streams must never share an instance — and getting a fresh
copy per stream, cheaply, is the orchestrator's job
(`PipelineOrchestrator._load_vad_frame_model`). What's left in this class is the
configuration those copies are driven with, and the per-stream reset hook.

An earlier version of this file also implemented segment-level streaming
detection. Nothing ever called it, and it ran the model and discarded the
result; it was removed rather than left to mislead the next reader.
"""
from fusion_runtime.vad.base import VADBase


class SileroVAD(VADBase):
    """Frame-level Silero detection: thresholds and sample rate live here."""

    def __init__(self, config):
        super().__init__(config)
        self._state = None

    async def reset(self):
        self._state = None
