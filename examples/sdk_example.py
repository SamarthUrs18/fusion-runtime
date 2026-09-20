#!/usr/bin/env python3
"""
Example: Using fusion-runtime as a Python library (SDK style).
"""
import asyncio

import numpy as np
from fusion_runtime import (
    LLMConfig,
    PipelineConfig,
    PipelineOrchestrator,
    STTConfig,
    TTSConfig,
    TurnDetectionConfig,
    run_single_turn,
)


async def main():
    # Models by catalog id (see `frun models list`); download them with `frun models pull`.
    config = PipelineConfig(
        stt=STTConfig(model="whisper-tiny.en", language="en"),
        llm=LLMConfig(runtime="llama_cpp", model="qwen2.5-0.5b-q4", n_ctx=2048, max_tokens=256),
        tts=TTSConfig(model="kokoro-v1.0", voice="af_heart"),
        # When the caller has finished: answer after 500 ms of silence; talking over the agent for
        # 300 ms interrupts it. For shorter waits on
        # finished sentences and longer ones mid-thought, plug in a turn detector model, e.g.
        #   TurnDetectionConfig(runtime="my_package.turns:MyDetector", model="...", min_silence_ms=500)
        # (see examples/turn_detector_plugin.py).
        turn_detection=TurnDetectionConfig(min_silence_ms=500, barge_in_min_speech_ms=300, resume_window_ms=1500),
        target_latency_ms=500,
    )

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()

    try:
        # Example 1: Single turn (complete audio in/out)
        print("🎤 Single-turn example...")

        # Generate test audio (1 second of silence)
        test_audio = (np.zeros(16000, dtype=np.int16)).tobytes()

        response_audio = await run_single_turn(
            orchestrator,
            test_audio,
            system_prompt="You are a helpful voice assistant. Keep responses very brief."
        )

        print(f"✅ Got response: {len(response_audio)} bytes audio")

        # Save for verification
        import soundfile as sf
        sf.write("output_response.wav", np.frombuffer(response_audio, dtype=np.int16), 24000)
        print("💾 Saved to output_response.wav")

        # Example 2: Streaming pipeline
        print("\n🌊 Streaming example...")

        async def audio_stream():
            # Simulate streaming audio chunks (100ms each)
            for _ in range(10):
                chunk = (np.zeros(1600, dtype=np.int16)).tobytes()
                yield chunk
                await asyncio.sleep(0.1)

        chunk_count = 0
        async for audio_chunk in orchestrator.run_pipeline(
            audio_stream(),
            system_prompt="You are a helpful voice assistant."
        ):
            chunk_count += 1
            if chunk_count <= 3:
                print(f"  📦 Chunk {chunk_count}: {len(audio_chunk)} bytes")

        print(f"✅ Streaming complete: {chunk_count} chunks")

        # Show metrics
        metrics = orchestrator.get_metrics_summary()
        print(f"\n📊 Metrics: {metrics}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
