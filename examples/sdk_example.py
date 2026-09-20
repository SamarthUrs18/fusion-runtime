#!/usr/bin/env python3
"""Running a voice turn inside your own process, with no server.

`frun up` is a thin wrapper around what this file does. Use the library directly
when the turn belongs inside something you already run: a queue worker, a test,
a batch job over recorded calls.

    python3 examples/sdk_example.py

Needs the development models (`frun models pull`) and a WAV of someone speaking,
16 kHz mono 16-bit. It uses the test fixture by default; pass your own:

    python3 examples/sdk_example.py my-recording.wav

Feeding it silence is not a smaller version of this example — speech-to-text
returns nothing for silence and the agent has nothing to answer, so the run
looks broken when it is working correctly.
"""
import asyncio
import sys
import wave
from pathlib import Path

from fusion_runtime import LLM, STT, TTS, Agent, PipelineOrchestrator, Turns, run_single_turn

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hello.wav"
TTS_SAMPLE_RATE = 24000  # what Kokoro produces

agent = Agent(
    name="shopkart-orders",
    prompt="You are the order line for ShopKart. Answer in one short sentence.",
    stt=STT("whisper-tiny.en"),
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=60),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    # Answer after 500 ms of silence; 300 ms of the caller talking over the agent
    # interrupts it. A turn-detector model can replace the fixed wait with one that
    # depends on whether the sentence sounded finished — see turn_detector_plugin.py.
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
)


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(f"{path} must be 16 kHz mono 16-bit; got "
                             f"{wav.getframerate()} Hz, {wav.getnchannels()}ch, "
                             f"{wav.getsampwidth() * 8}-bit")
        return wav.readframes(wav.getnframes())


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TTS_SAMPLE_RATE)
        wav.writeframes(pcm)


async def main(source: Path) -> None:
    audio = read_wav(source)
    print(f"caller audio: {len(audio) / 2 / 16000:.2f}s from {source.name}")

    # initialize() loads three models and is the expensive call — seconds, not
    # milliseconds. Hold the orchestrator and reuse it across turns.
    orchestrator = PipelineOrchestrator(agent.config())
    await orchestrator.initialize()

    try:
        # The whole reply, once it's finished.
        reply = await run_single_turn(orchestrator, audio, system_prompt=agent.prompt)
        out = Path("reply.wav")
        write_wav(out, reply)
        print(f"reply: {len(reply) / 2 / TTS_SAMPLE_RATE:.2f}s of speech, written to {out}")

        # Or the same turn as it is produced. This is the one that matters for a
        # live call: the first chunk arrives while the model is still generating,
        # which is what makes a reply start playing in under half a second.
        async def in_chunks(pcm: bytes, ms: int = 100):
            step = 16000 * 2 // 1000 * ms
            for i in range(0, len(pcm), step):
                yield pcm[i:i + step]
                await asyncio.sleep(ms / 1000)

        chunks = total = 0
        async for chunk in orchestrator.run_pipeline(in_chunks(audio), agent.prompt):
            chunks += 1
            total += len(chunk)
        # A chunk is roughly a sentence — text-to-speech is handed whole sentences so
        # it can get the prosody right — so a one-sentence reply is legitimately one
        # chunk. The count tells you how the reply was cut up, not how well it streamed.
        print(f"streamed: {chunks} chunk(s), {total / 2 / TTS_SAMPLE_RATE:.2f}s of speech")

        print(f"metrics: {orchestrator.get_metrics_summary()}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE))
