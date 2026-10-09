"""`frun bench`: how a running server behaves with several callers at once.

Opens N real WebSocket sessions against a running server, streams the same recording down each in
real time, and reports how long each caller waited for audio to come back, and where that time went.

    frun up agent.py --host 0.0.0.0        # in one terminal
    frun bench --callers 1,4,8,12          # in another

The number that matters is how the median moves from one caller to several: an in-process llama.cpp
model decodes one reply at a time, so callers queue; on vLLM or SGLang they don't, until speech does.
`--target-ms` turns that into a capacity: how many callers stayed under it. That's what a quote for a
customer's agent is based on, measured rather than promised.

Audio is streamed in real time on purpose: sending a recording at once would measure something the
runtime never does, and turn detection watches silence in wall-clock time.
"""
import asyncio
import json
import statistics
import time
import wave
from pathlib import Path
from typing import List, Optional

import typer

AUDIO_DIR = Path(__file__).resolve().parent / "bench_audio"
RECORDINGS = {"hello": "hello.wav", "order": "order_1042.wav"}  # order: asks about order 1042, for tool agents
CHUNK_MS = 20
# Where each reply's time went, from the server's own turn summary: which stage starts to queue as
# callers are added is the finding, not just that it's slower.
STAGES = {"stt_transcribe_ms": "stt", "llm_queue_ms": "llm queue", "llm_first_token_ms": "llm first token",
          "tool_ms": "tools", "tts_first_chunk_ms": "tts first audio"}


def load_audio(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise typer.BadParameter(f"{path} must be a 16 kHz mono 16-bit WAV")
        return wav.readframes(wav.getnframes())


def audio_path(name: str) -> Path:
    if name in RECORDINGS:
        return AUDIO_DIR / RECORDINGS[name]
    path = Path(name).expanduser()
    if not path.is_file():
        raise typer.BadParameter(f"no recording {name!r}: use {', '.join(RECORDINGS)} or a path to a WAV")
    return path


def resolve_key(url: str, given: Optional[str]) -> str:
    """The key to present, the way `frun talk` works it out.

    FUSION_API_KEY if it's set; otherwise, for a server on this machine only, the first key that
    server accepts. Never falls back for a remote URL: that would send your server's keys elsewhere.
    """
    if given:
        return given
    from fusion_runtime.env import load_env_file
    from fusion_runtime.security.keys import client_key

    load_env_file()
    http_url = url.replace("ws://", "http://").replace("wss://", "https://")
    return client_key(http_url) or ""


async def one_session(index: int, url: str, key: str, audio: bytes, turns: int, results: list) -> None:
    """One caller: connect, then take `turns` turns, timing each reply."""
    import websockets

    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:  # websockets renamed this parameter; support both without pinning a version
        ws = await websockets.connect(url, additional_headers=headers)
    except TypeError:
        ws = await websockets.connect(url, extra_headers=headers)

    step = 16000 * 2 // 1000 * CHUNK_MS
    try:
        await ws.recv()  # the config message
        # A real microphone never goes digitally silent, and fusion tells a caller sending exact
        # zeros that their mic is muted, so the quiet between turns is faint room noise.
        quiet = b"\x01\x00" * (step // 2)

        # The first turn allocates caches and isn't representative: taken and thrown away.
        for turn in range(-1, turns):
            for i in range(0, len(audio), step):
                await ws.send(audio[i:i + step])
                await asyncio.sleep(CHUNK_MS / 1000)
            stopped_talking_at = time.perf_counter()

            first_audio_at = None
            summary: dict = {}
            deadline = stopped_talking_at + 60

            async def keep_the_mic_open(deadline=deadline):
                # The turn detector ends a turn by hearing silence; stopping the stream would leave
                # the server waiting on a timeout and put that wait into every measurement.
                while time.perf_counter() < deadline:
                    await ws.send(quiet)
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
                        summary = event.get("summary") or {}
                        break  # the turn is over; the server said so
            finally:
                mic.cancel()

            if turn < 0:
                continue  # the warm-up turn
            if first_audio_at is None:
                results.append({"caller": index, "turn": turn, "error": "no audio came back"})
                continue
            results.append({
                "caller": index,
                "turn": turn,
                "wall_ms": round((first_audio_at - stopped_talking_at) * 1000),
                "server_ms": round(summary["response_ms"]) if summary.get("response_ms") else None,
                **{stage: summary.get(stage) for stage in STAGES},
                "tool_calls": summary.get("tool_calls") or 0,
            })
    finally:
        await ws.close()


async def run_round(callers: int, url: str, key: str, audio: bytes, turns: int) -> list:
    results: list = []
    await asyncio.gather(*[one_session(i, url, key, audio, turns, results) for i in range(callers)])
    return results


def summarize(callers: int, results: list, seconds: float) -> dict:
    """One round as numbers: what `report` prints and `--json` saves."""
    ok = [r for r in results if "wall_ms" in r]
    errors = [r for r in results if "error" in r]
    row: dict = {"callers": callers, "turns": len(ok), "seconds": round(seconds, 1), "failed": len(errors),
                 "error": errors[0].get("error") if errors else None}
    if ok:
        walls = sorted(r["wall_ms"] for r in ok)
        row.update(median_ms=round(statistics.median(walls)), worst_ms=walls[-1])
        server = [r["server_ms"] for r in ok if r["server_ms"]]
        row["server_median_ms"] = round(statistics.median(server)) if server else None
        row["stages_median_ms"] = {label: round(statistics.median(values)) for stage, label in STAGES.items()
                                   if (values := [r[stage] for r in ok if r.get(stage) is not None])}
        row["tool_calls"] = sum(r["tool_calls"] for r in ok)
    return row


def report(row: dict, baseline: Optional[float]) -> str:
    if "median_ms" not in row:
        return f"  {row['callers']:>2} caller(s): nothing completed. {row['error'] or ''}".rstrip()
    line = (f"  {row['callers']:>2} caller(s):  median {row['median_ms']:>5} ms   worst {row['worst_ms']:>5} ms   "
            f"{row['turns']} turns in {row['seconds']}s")
    if row["server_median_ms"]:
        line += f"   (server said {row['server_median_ms']} ms)"
    if baseline:
        line += f"   {row['median_ms'] / baseline:.1f}x of one caller"
    lines = [line]
    if row["stages_median_ms"]:
        lines.append("      median ms: " + ", ".join(f"{k} {v}" for k, v in row["stages_median_ms"].items()))
    if row["tool_calls"]:
        lines.append(f"      {row['tool_calls']} tool call(s) in {row['turns']} turns")
    if row["failed"]:
        lines.append(f"      {row['failed']} turn(s) failed: {row['error']}")
    return "\n".join(lines)


def capacity(rows: List[dict], target_ms: float) -> str:
    """The most callers whose median stayed under the target, with every round below it clean too."""
    best = None
    for row in sorted(rows, key=lambda r: r["callers"]):
        if row.get("median_ms") is None or row["failed"] or row["median_ms"] > target_ms:
            break
        best = row
    if best is None:
        return f"No round stayed under {target_ms:.0f} ms (median) without failures."
    return (f"Up to {best['callers']} caller(s) stayed under {target_ms:.0f} ms (median {best['median_ms']} ms, "
            f"worst {best['worst_ms']} ms) on this setup.")


async def _bench(url: str, key: str, audio: bytes, callers: List[int], turns: int, label: str,
                 target_ms: Optional[float], json_out: Optional[Path]) -> None:
    ws_url = url.rstrip("/") + "/v1/voice/ws"
    typer.echo(f"{label or ws_url}  —  {len(audio) / 2 / 16000:.2f}s of caller audio, {turns} turn(s) each\n")
    rows, baseline = [], None
    for count in callers:
        started = time.perf_counter()
        results = await run_round(count, ws_url, key, audio, turns)
        row = summarize(count, results, time.perf_counter() - started)
        rows.append(row)
        typer.echo(report(row, baseline))
        if count == 1 and row.get("median_ms"):
            baseline = row["median_ms"]
        await asyncio.sleep(2)  # let sessions close before the next round claims slots
    typer.echo("")
    if target_ms:
        typer.echo(capacity(rows, target_ms))
    else:
        typer.echo("A median that climbs in step with the caller count means replies are queueing.\n"
                   "Put the agent's LLM on vLLM or SGLang and run this again to see what changes;\n"
                   "--target-ms 1500 turns the rounds into a capacity.")
    if json_out:
        json_out.write_text(json.dumps({"label": label or ws_url, "turns_per_caller": turns,
                                        "target_ms": target_ms, "rounds": rows}, indent=2) + "\n")
        typer.echo(f"Saved: {json_out}")


def bench(
    url: str = typer.Option("ws://127.0.0.1:8000", "--url", help="The running server."),
    key: str = typer.Option(None, "--key", help="An accepted key. For a server on this machine it's found for you."),
    callers: str = typer.Option("1,2,4", "--callers", help="How many callers at once, per round, comma separated."),
    turns: int = typer.Option(3, "--turns", min=1, help="Turns per caller, after one warm-up turn."),
    audio: str = typer.Option("hello", "--audio",
                              help="What callers say: hello, order (asks about order 1042, for tool agents), "
                                   "or a 16 kHz mono 16-bit WAV."),
    target_ms: float = typer.Option(None, "--target-ms", min=1,
                                    help="Report how many callers stayed under this median wait (e.g. 1500)."),
    json_out: Path = typer.Option(None, "--json", help="Also save every round's numbers to this file."),
    label: str = typer.Option("", "--label", help="A heading for the run, e.g. 'RTX 4090, vLLM'."),
) -> None:
    """Measure a running server with several callers at once: wait per caller, and where it went."""
    try:
        counts = [int(n) for n in callers.split(",") if n.strip()]
    except ValueError:
        raise typer.BadParameter(f"--callers takes numbers separated by commas, like 1,4,8; got {callers!r}") from None
    if not counts or min(counts) < 1:
        raise typer.BadParameter("--callers needs at least one number, each 1 or more")
    sound = load_audio(audio_path(audio))
    try:
        asyncio.run(_bench(url, resolve_key(url, key), sound, counts, turns, label, target_ms, json_out))
    except OSError as e:
        typer.echo(f"Error: can't reach {url} ({e}). Is the server running? Start it with: frun up", err=True)
        raise typer.Exit(1)
