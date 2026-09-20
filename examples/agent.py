#!/usr/bin/env python3
"""A voice agent in one file. Run it with:

    frun up examples/agent.py            # add --reload to restart when you edit it
    frun talk                            # in another terminal

Everything the agent is lives here: what it says, which models it uses, and how
it takes turns. Secrets don't: those come from environment variables.
"""
from fusion_runtime import LLM, STT, TTS, VAD, Agent, Turns

agent = Agent(
    name="shopkart-orders",
    prompt=(
        "You are the order line for ShopKart, an online store. "
        "Help callers check an order, change a delivery address, or start a return. "
        "Keep every answer to one short sentence, and ask for the order number when you need it."
    ),
    # Models: a catalog id (`frun models list`), a file path, hf:owner/repo, or a URL.
    # A plain name is enough when there's nothing to configure.
    stt=STT("whisper-tiny.en"),
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=200, temperature=0.6),
    tts=TTS("kokoro-v1.0", voice="af_heart", speed=1.0),
    language="en",
    # Turn taking: how long a caller may pause before the agent answers, and how
    # long they must talk over it before it stops.
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
    # A turn detector model can replace the fixed wait — see examples/turn_detector_plugin.py:
    # turns=Turns("examples.turn_detector_plugin:TrailingWordsDetector", wait_ms=500),
    # Which audio counts as speech; raise the threshold in a noisy room.
    vad=VAD(threshold=0.5),
)
