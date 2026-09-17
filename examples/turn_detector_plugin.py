#!/usr/bin/env python3
"""Example: plugging your own turn detector into fusion-runtime.

By default the agent answers after a fixed pause (TurnDetectionConfig.min_silence_ms).
A turn detector makes that pause smarter: it says how likely the caller is done,
and the engine waits less when they are and longer when they're mid-thought.

Any model works: a text end-of-utterance classifier, an audio prosody model, or
a service. Implement `predict`, declare what you need (`uses_audio`,
`uses_history`), and point config at the class. Check it with the conformance kit
before using it on calls:

    python examples/turn_detector_plugin.py

Use it with the server (from the repository root, so Python can import it):

    PYTHONPATH=. frun up --turn-detector examples.turn_detector_plugin:TrailingWordsDetector

or in Python:

    PipelineConfig(turn_detection=TurnDetectionConfig(
        runtime="examples.turn_detector_plugin:TrailingWordsDetector",
    ))

or package it and register an entry point, so config can say runtime="trailing_words":

    [project.entry-points."fusion_runtime.runtimes"]
    "turn.trailing_words" = "my_package.turns:TrailingWordsDetector"

Only plug in models whose license allows your use.
"""
import asyncio
import re

from fusion_runtime.contract import Capabilities, InvalidRequest, TurnDetector, TurnPrediction, TurnRequest

# An English caller who stops on one of these words is usually not done:
# "I'd like a table for ... four", "and ... also a window seat".
UNFINISHED_ENDINGS = {
    "a", "an", "the", "and", "or", "but", "so", "because", "for", "to", "of", "with", "at", "in", "on",
    "my", "your", "is", "are", "was", "i", "i'm", "um", "uh", "like", "about", "if", "then",
}


class TrailingWordsDetector(TurnDetector):
    """A tiny heuristic detector (English only), to show the shape of a real one."""

    uses_audio = False
    uses_history = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming_output=False, max_concurrency=64, languages=("en",))

    async def load(self) -> None:
        # A model-based detector loads its weights here, off the event loop:
        #     self.model = await asyncio.get_running_loop().run_in_executor(None, load_my_model, self.spec.model)
        pass

    async def predict(self, request: TurnRequest) -> TurnPrediction:
        request.cancel.raise_if_cancelled()
        text = request.transcript.strip()
        if not text:
            raise InvalidRequest("transcript must not be empty")
        # A model-based detector runs inference in a thread, for example:
        #     p = await asyncio.get_running_loop().run_in_executor(None, self.model.score, text, request.history)
        last_word = re.sub(r"[^\w']", " ", text.lower()).split()[-1:] or [""]
        if last_word[0] in UNFINISHED_ENDINGS:
            return TurnPrediction(0.05, extra={"trailing_word": last_word[0]})
        return TurnPrediction(0.5)  # unsure: the engine keeps its normal wait


async def main() -> None:
    from fusion_runtime.contract import ModelSpec
    from fusion_runtime.testing.conformance import check_runtime

    detector = TrailingWordsDetector(ModelSpec(stage="turn", runtime="trailing_words", model=""))
    print(await check_runtime(detector))


if __name__ == "__main__":
    asyncio.run(main())
