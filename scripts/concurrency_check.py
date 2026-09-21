#!/usr/bin/env python3
"""How the runtime behaves with more than one caller at a time.

Opens N real WebSocket sessions against a running server, streams the same
recording down each in real time, and reports how long each caller waited for
audio to come back. Run it against a server you started yourself:

    frun up --host 0.0.0.0                       # in one terminal
    python3 scripts/concurrency_check.py --key frun_...   # in another

    python3 scripts/concurrency_check.py --callers 1,2,4 --turns 3
    python3 scripts/concurrency_check.py --url ws://localhost:8000 --label "llama-server -np 4"

The number that matters is how the median moves from one caller to several. An
in-process llama.cpp context decodes one reply at a time, so callers queue
behind each other; pointing the agent's LLM at vLLM or `llama-server -np 4`
is what changes that, and running this twice is how you show it rather than
claim it.

Audio is streamed in real time on purpose. Sending a whole recording at once
would measure something the runtime never does, and turn detection watches
silence in wall-clock time.
"""
import argparse
import asyncio
import json
import statistics
import time
import wave
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hello.wav"
CHUNK_MS = 20


def load_audio(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(f"{path} must be 16 kHz mono 16-bit")
        return wav.readframes(wav.getnframes())


def resolve_key(url: str, given: str) -> str:
    """The key to present, the same way `frun talk` works it out.

    FUSION_API_KEY if it's set; otherwise, for a server on this machine only, the
    first key that server accepts. Never falls back for a remote URL — that would
    send your server's keys to someone else's.
    """
    if given:
        return given
    from fusion_runtime.env import load_env_file
    from fusion_runtime.security.keys import client_key

    load_env_file()
    http_url = url.replace("ws://", "http://").replace("wss://", "https://")
    return client_key(http_url) or ""


async def one_session(index: int, url: str, key: str, audio: bytes, turns: int, results: list):
    """One caller: connect, then take `turns` turns, timing each reply."""
    import websockets

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # websockets renamed this parameter; support both without pinning a version.
    try:
        ws = await websockets.connect(url, additional_headers=headers)
    except TypeError:
        ws = await websockets.connect(url, extra_headers=headers)

    step = 16000 * 2 // 1000 * CHUNK_MS
    try:
        await ws.recv()  # the config message

        silence = bytes(step)

        # The first turn of a session allocates caches and is not representative,
        # so one is taken and thrown away before anything is measured.
        for turn in range(-1, turns):
            for i in range(0, len(audio), step):
                await ws.send(audio[i:i + step])
                await asyncio.sleep(CHUNK_MS / 1000)
            stopped_talking_at = time.perf_counter()

            first_audio_at = None
            server_ms = None
            deadline = stopped_talking_at + 60

            async def keep_the_mic_open(deadline=deadline):
                # A real microphone doesn't go quiet when the caller stops speaking,
                # and the turn detector ends a turn by hearing silence. Stopping the
                # stream here would leave the server waiting on a timeout and put
                # that wait into every measurement.
                while time.perf_counter() < deadline:
                    await ws.send(silence)
                    await asyncio.sleep(CHUNK_MS / 1000)

            mic = asyncio.create_task(keep_the_mic_open())
            try:
                while time.perf_counter() < deadline:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.perf_counter()))
                    except asyncio.TimeoutError:
                        break
                    if isinstance(message, bytes):
                        if first_audio_at is None:
                            first_audio_at = time.perf_counter()
                        continue
                    event = json.loads(message)
                    if event.get("type") == "error":
                        results.append({"caller": index, "turn": turn, "error": event.get("message")})
                        return
                    if event.get("type") == "turn.trace":
                        server_ms = (event.get("summary") or {}).get("response_ms")
                        break  # the turn is over; the server said so
            finally:
                mic.cancel()

            if turn < 0:
                continue  # the warm-up
            if first_audio_at is None:
                results.append({"caller": index, "turn": turn, "error": "no audio came back"})
                continue
            results.append({
                "caller": index,
                "turn": turn,
                "wall_ms": round((first_audio_at - stopped_talking_at) * 1000),
                "server_ms": round(server_ms) if server_ms else None,
            })
    finally:
        await ws.close()


async def run_round(callers: int, url: str, key: str, audio: bytes, turns: int) -> list:
    results: list = []
    await asyncio.gather(*[
        one_session(i, url, key, audio, turns, results) for i in range(callers)
    ])
    return results


def report(callers: int, results: list, seconds: float, baseline: float = None) -> float:
    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "wall_ms" in r]
    if not ok:
        print(f"  {callers} caller(s): nothing completed. {errors[:2]}")
        return None
    walls = sorted(r["wall_ms"] for r in ok)
    median = statistics.median(walls)
    worst = walls[-1]
    server = [r["server_ms"] for r in ok if r["server_ms"]]
    line = (f"  {callers:>2} caller(s):  median {median:>5.0f} ms   worst {worst:>5.0f} ms   "
            f"{len(ok)} turns in {seconds:.1f}s")
    if server:
        line += f"   (server said {statistics.median(server):.0f} ms)"
    if baseline:
        line += f"   {median / baseline:.1f}x of one caller"
    print(line)
    if errors:
        print(f"      {len(errors)} turn(s) failed: {errors[0].get('error')}")
    return median


async def main(args) -> None:
    audio = load_audio(Path(args.audio) if args.audio else FIXTURE)
    key = resolve_key(args.url, args.key)
    url = args.url.rstrip("/") + "/v1/voice/ws"
    print(f"{args.label or url}  —  {len(audio) / 2 / 16000:.2f}s of caller audio, "
          f"{args.turns} turn(s) each\n")

    baseline = None
    for callers in [int(n) for n in args.callers.split(",")]:
        started = time.perf_counter()
        results = await run_round(callers, url, key, audio, args.turns)
        median = report(callers, results, time.perf_counter() - started, baseline)
        if callers == 1:
            baseline = median
        await asyncio.sleep(2)  # let sessions close before the next round claims slots

    print("\nA median that climbs roughly in step with the caller count means replies are\n"
          "queueing behind one another. Point the agent's LLM at vLLM or `llama-server -np 4`\n"
          "and run this again to see what changes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://localhost:8000", help="the running server")
    p.add_argument("--key", default="", help="an accepted API key; for a server on this machine it is found for you")
    p.add_argument("--callers", default="1,2,4", help="how many at once, comma separated")
    p.add_argument("--turns", type=int, default=3, help="turns per caller")
    p.add_argument("--audio", help="a 16 kHz mono 16-bit WAV (default: the test fixture)")
    p.add_argument("--label", help="what to print as the heading, e.g. 'in-process llama.cpp'")
    asyncio.run(main(p.parse_args()))
