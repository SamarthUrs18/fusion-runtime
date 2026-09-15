"""Turn detection: deciding when the user has finished speaking."""
from abc import ABC
from dataclasses import dataclass


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


class TurnDetectorBase(ABC):
    """Base class for End-of-Turn *content* detection.

    This answers exactly one question — "does this accumulated transcript
    read like a finished utterance on its own?" — and nothing about timing.
    Silence timing is handled separately, by an always-running watcher in
    `PipelineOrchestrator._llm_stage` (see its docstring for why this had
    to be split out: reacting to silence only when a *new* STT result
    arrives means the silence being measured is the pause *before* that
    new speech, not a pause *after* the user actually stopped talking —
    which is backwards, and was cutting turns off mid-sentence on any
    ordinary thinking-pause). The watcher uses this signal only to pick
    between a short confirmation pause (confidently complete) and a
    longer, safer one (ambiguous) — it doesn't own the decision.
    """

    def __init__(self, config):
        self.config = config

    def looks_complete(self, text: str) -> bool:
        """Default heuristic: ends with sentence-ending punctuation.
        Subclasses (e.g. a real EoT model) can override with something
        smarter — text + prosody, not just the trailing character."""
        return text.strip().endswith((".", "!", "?", "。", "！", "？"))

    async def reset(self):
        pass


class PunctuationTurnDetector(TurnDetectorBase):
    """Text ending in sentence punctuation counts as a finished thought, so a
    shorter pause ends the turn. Uses TurnDetectorBase's `looks_complete`."""
    pass
