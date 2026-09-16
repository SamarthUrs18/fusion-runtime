#!/usr/bin/env python3
"""Latency guardrail: time to first audio, and how long the event loop gets blocked.

Run before and after any change to the engine or a runtime, on the same machine:

    python3 scripts/bench_latency.py            # development profile, 6 measured turns
    python3 scripts/bench_latency.py --turns 10

Reference on an 8 GB M1 MacBook Air (development profile, CPU), Sep 2026:
first-audio median ~660 ms; event loop blocked at most ~25 ms while the LLM streams.
Numbers vary by a few tens of ms between runs; close other heavy apps first.
"""
import argparse
import asyncio
import statistics
import time
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hello.wav"


async def first_audio_latency(orchestrator, audio: bytes, turns: int) -> list[int]:
    """Turn end detected → first reply audio, per turn, read from telemetry (turn.summary.response_ms)."""
    from fusion_runtime import run_single_turn
    from fusion_runtime.telemetry import ListSink, telemetry

    sink = ListSink()
    telemetry.add_sink(sink)
    try:
        # One warm-up turn that isn't counted: first calls allocate caches.
        await run_single_turn(orchestrator, audio, system_prompt="Reply in one short sentence.")
        sink.events.clear()
        for _ in range(turns):
            await run_single_turn(orchestrator, audio, system_prompt="Reply in one short sentence.")
    finally:
        telemetry.remove_sink(sink)
    summaries = [e.attrs for e in sink.named("turn.summary")]
    interrupted = [s for s in summaries if s.get("outcome") != "completed"]
    if interrupted:
        print(f"warning: {len(interrupted)} turn(s) didn't complete normally: {[s.get('outcome') for s in interrupted]}")
    return [round(s["response_ms"]) for s in summaries if s.get("response_ms") is not None]


async def event_loop_lag(llm) -> tuple[float, float, float]:
    from fusion_runtime.llm import ChatMessage

    gaps, running = [], True

    async def heartbeat():
        last = time.perf_counter()
        while running:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append((now - last) * 1000)
            last = now

    beat = asyncio.create_task(heartbeat())
    messages = [
        ChatMessage("system", "Answer in about three sentences."),
        ChatMessage("user", "Describe a busy train station."),
    ]
    async for _ in llm.generate_stream(messages):
        pass
    running = False
    await beat
    gaps.sort()
    return statistics.median(gaps), gaps[int(len(gaps) * 0.95)], gaps[-1]


async def main(turns: int) -> None:
    from fusion_runtime import DEVELOPMENT_CONFIG, PipelineOrchestrator

    from fusion_runtime.telemetry import telemetry

    telemetry.configure(format="off")  # numbers only; run `frun up` to see the full event stream
    orchestrator = PipelineOrchestrator(DEVELOPMENT_CONFIG)
    await orchestrator.initialize()
    audio = FIXTURE.read_bytes()[44:]  # skip the WAV header

    latencies = await first_audio_latency(orchestrator, audio, turns)
    median, p95, worst = await event_loop_lag(orchestrator.llm)
    await orchestrator.shutdown()

    print(f"first audio after turn end (ms): {latencies}  median={statistics.median(latencies):.0f}")
    print(f"event loop blocked (ms, ideal ~5): median={median:.1f}  p95={p95:.1f}  max={worst:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=6, help="measured turns (after one warm-up turn)")
    asyncio.run(main(parser.parse_args().turns))
