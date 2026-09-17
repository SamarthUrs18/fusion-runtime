"""Turn detection runtime contract: has the user finished their turn?

Silence alone is a poor signal: people pause mid-sentence ("I'd like a table
for... four"), and waiting long enough to be safe makes every reply feel slow.
A turn detector looks at what was said (and, if it wants, the conversation so
far and the audio) and returns how likely it is that the user is done. The
engine turns that into how much silence to wait for:

    likely done        (>= likely_threshold)     short wait   (min_confident_silence_ms)
    unsure                                       normal wait  (min_silence_ms)
    likely mid-thought (< unlikely_threshold)    long wait    (max_silence_ms)

So a detector never ends a turn by itself: silence still has to confirm it,
and a slow or failing detector only leaves the default wait in place; it
can never hang a call.

Detectors can be text models (end-of-utterance classifiers over the
transcript and recent messages), audio models (prosody: pitch and pace at
the end of speech), heuristics, or a call to a service. Declare what you use
with `uses_audio` / `uses_history` so the engine only collects what's needed.
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

from fusion_runtime.contract.common import ModelRuntime, Request
from fusion_runtime.contract.llm import Message


@dataclass(kw_only=True)
class TurnRequest(Request):
    transcript: str  # what the user has said this turn so far (latest cumulative transcript)
    history: Sequence[Message] = ()  # earlier messages in the call, oldest first (when uses_history)
    audio: Optional[bytes] = None  # this turn's speech, 16-bit mono PCM (when uses_audio), most recent last
    sample_rate: int = 16000
    silence_ms: float = 0.0  # silence since the user last spoke, for detectors that weigh it themselves


@dataclass
class TurnPrediction:
    end_of_turn: float  # probability in [0, 1] that the user has finished speaking
    extra: dict = field(default_factory=dict)  # detector-specific details, for logs


class TurnDetector(ModelRuntime):
    stage = "turn"
    uses_audio: bool = False
    uses_history: bool = False
    refines_wait: bool = True  # False for detectors whose answer never changes (the built-in silence detector)

    @abstractmethod
    async def predict(self, request: TurnRequest) -> TurnPrediction:
        """How likely is it that the user has finished their turn?

        Must not block the event loop (run model inference in a thread).
        Should answer in well under 100 ms; the engine keeps using the last
        answer while a new one is computed. Raise Cancelled if the request
        is cancelled, InvalidRequest for an empty transcript.
        """
