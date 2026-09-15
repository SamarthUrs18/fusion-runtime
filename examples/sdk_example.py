#!/usr/bin/env python3
"""
Example: Using fusion-runtime as a Python library (SDK style).
"""
import asyncio
import numpy as np
from fusion_runtime import (
    PipelineConfig,
    PipelineOrchestrator,
    STTConfig,
    LLMConfig,
    TTSConfig,
    Provider,
    run_single_turn,
)


async def main():
    # Create config (customize providers as needed)
    config = PipelineConfig(
        stt=STTConfig(
            provider=Provider.FASTER_WHISPER,
            model="tiny.en",
            device="cuda",
            compute_type="float16",
        ),
        llm=LLMConfig(
            provider=Provider.LLAMA_CPP,
            model="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            n_gpu_layers=-1,
        ),
        tts=TTSConfig(
            provider=Provider.KOKORO,
            model="kokoro-v1.0.onnx",
            voice="af_heart",
        ),
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
            for i in range(10):
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