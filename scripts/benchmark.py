#!/usr/bin/env python3
"""
Benchmark script to measure fusion-runtime latency.
Run after models are downloaded and server is running.
"""
import asyncio
import base64
import time
import statistics
import httpx
import numpy as np
from pathlib import Path


async def benchmark_rest(num_requests: int = 10, url: str = "http://localhost:8000/v1/voice/chat"):
    """Benchmark REST endpoint."""
    print(f"🔬 Benchmarking REST: {num_requests} requests...")
    
    # Create test audio (1 second silence)
    test_audio = (np.zeros(16000, dtype=np.int16)).tobytes()
    audio_b64 = base64.b64encode(test_audio).decode()
    
    latencies = []
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Warmup
        for _ in range(2):
            await client.post(url, json={"audio_base64": audio_b64})
        
        # Benchmark
        for i in range(num_requests):
            start = time.perf_counter()
            response = await client.post(url, json={"audio_base64": audio_b64})
            latency = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(latency)
                print(f"  Request {i+1}: {latency:.0f}ms")
            else:
                print(f"  Request {i+1}: FAILED ({response.status_code})")
    
    if latencies:
        print(f"\n📊 Results ({len(latencies)} successful):")
        print(f"  Mean:   {statistics.mean(latencies):.0f}ms")
        print(f"  Median: {statistics.median(latencies):.0f}ms")
        print(f"  P95:    {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
        print(f"  P99:    {max(latencies):.0f}ms")
        print(f"  Min:    {min(latencies):.0f}ms")
        print(f"  Max:    {max(latencies):.0f}ms")


async def benchmark_streaming(num_requests: int = 5, url: str = "http://localhost:8000/v1/voice/stream"):
    """Benchmark streaming endpoint - measures time to first audio chunk."""
    print(f"🔬 Benchmarking Streaming (TTFA): {num_requests} requests...")
    
    test_audio = (np.zeros(16000, dtype=np.int16)).tobytes()
    audio_b64 = base64.b64encode(test_audio).decode()
    
    ttfa_latencies = []  # Time to first audio
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(num_requests):
            start = time.perf_counter()
            
            async with client.stream("POST", url, json={"audio_base64": audio_b64}) as response:
                if response.status_code == 200:
                    # Read first chunk
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            ttfa = (time.perf_counter() - start) * 1000
                            ttfa_latencies.append(ttfa)
                            print(f"  Request {i+1} TTFA: {ttfa:.0f}ms")
                            break
                else:
                    print(f"  Request {i+1}: FAILED ({response.status_code})")
    
    if ttfa_latencies:
        print(f"\n📊 Time to First Audio ({len(ttfa_latencies)} successful):")
        print(f"  Mean:   {statistics.mean(ttfa_latencies):.0f}ms")
        print(f"  Median: {statistics.median(ttfa_latencies):.0f}ms")
        print(f"  P95:    {statistics.quantiles(ttfa_latencies, n=20)[18]:.0f}ms")
        print(f"  Min:    {min(ttfa_latencies):.0f}ms")
        print(f"  Max:    {max(ttfa_latencies):.0f}ms")


async def benchmark_sdk(num_runs: int = 5):
    """Benchmark using SDK directly (no HTTP overhead)."""
    print(f"🔬 Benchmarking SDK (direct): {num_runs} runs...")
    
    from fusion_runtime import PipelineConfig, PipelineOrchestrator, STTConfig, LLMConfig, TTSConfig, Provider, run_single_turn
    
    config = PipelineConfig(
        stt=STTConfig(provider=Provider.FASTER_WHISPER, model="tiny.en", device="cuda"),
        llm=LLMConfig(provider=Provider.LLAMA_CPP, model="Qwen2.5-7B-Instruct-Q4_K_M.gguf", n_gpu_layers=-1),
        tts=TTSConfig(provider=Provider.KOKORO, model="kokoro-v1.0.onnx"),
    )
    
    orchestrator = PipelineOrchestrator(config)
    await orchestrator.initialize()
    
    test_audio = (np.zeros(16000, dtype=np.int16)).tobytes()
    
    latencies = []
    
    try:
        # Warmup
        for _ in range(2):
            await run_single_turn(orchestrator, test_audio)
        
        # Benchmark
        for i in range(num_runs):
            start = time.perf_counter()
            await run_single_turn(orchestrator, test_audio)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            print(f"  Run {i+1}: {latency:.0f}ms")
        
        if latencies:
            print(f"\n📊 Direct SDK Results:")
            print(f"  Mean:   {statistics.mean(latencies):.0f}ms")
            print(f"  Median: {statistics.median(latencies):.0f}ms")
            print(f"  P95:    {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
            print(f"  Min:    {min(latencies):.0f}ms")
            print(f"  Max:    {max(latencies):.0f}ms")
            
            # Show detailed metrics
            metrics = orchestrator.get_metrics_summary()
            print(f"\n  Pipeline breakdown (P50):")
            print(f"    STT:      {metrics.get('stt_p50', 0):.0f}ms")
            print(f"    LLM first: {metrics.get('llm_first_p50', 0):.0f}ms")
            print(f"    TTS first: {metrics.get('tts_first_p50', 0):.0f}ms")
            print(f"    E2E:       {metrics.get('e2e_p50', 0):.0f}ms")
    
    finally:
        await orchestrator.shutdown()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark fusion-runtime")
    parser.add_argument("--rest", action="store_true", help="Benchmark REST API")
    parser.add_argument("--stream", action="store_true", help="Benchmark streaming (TTFA)")
    parser.add_argument("--sdk", action="store_true", help="Benchmark SDK directly")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of requests")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    
    if not any([args.rest, args.stream, args.sdk, args.all]):
        args.all = True
    
    print("🚀 fusion-runtime Benchmark")
    print("=" * 50)
    
    if args.all or args.sdk:
        await benchmark_sdk(args.num)
        print()
    
    if args.all or args.rest:
        await benchmark_rest(args.num, f"{args.url}/v1/voice/chat")
        print()
    
    if args.all or args.stream:
        await benchmark_streaming(args.num, f"{args.url}/v1/voice/stream")
        print()
    
    print("✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())