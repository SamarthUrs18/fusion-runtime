"""The default turn detector: the turn ends after a pause, however it sounds.

It makes no prediction, so the engine waits `min_silence_ms` of silence
before the agent speaks. Predictable, language-independent and never fooled
by the punctuation Whisper adds to partial transcripts ("Hello." after a
greeting, mid-sentence). For shorter waits on finished sentences and longer
ones mid-thought, plug in a turn detector model.
"""
from fusion_runtime.contract import Capabilities, InvalidRequest, TurnDetector, TurnPrediction, TurnRequest

UNSURE = 0.5


class SilenceTurnDetector(TurnDetector):
    refines_wait = False  # the engine needn't ask: the answer is always "unsure"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming_output=False, max_concurrency=64)

    async def load(self) -> None:
        pass

    async def predict(self, request: TurnRequest) -> TurnPrediction:
        request.cancel.raise_if_cancelled()
        if not request.transcript.strip():
            raise InvalidRequest("transcript must not be empty")
        return TurnPrediction(UNSURE)
