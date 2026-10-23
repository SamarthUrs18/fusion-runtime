#!/usr/bin/env python3
"""Call a fusion agent the way a phone call through Twilio would, from this machine.

    frun up examples/tools_agent.py                       # with FUSION_TWILIO_AUTH_TOKEN set
    python3 tools/phone_sim/phone_sim.py                  # talk with your microphone
    python3 tools/phone_sim/phone_sim.py --say question.wav --record agent.wav

Everything Twilio does, nothing it doesn't: a signed webhook, the TwiML reply, the stream, 8 kHz
μ-law both ways, marks once audio has played, clear when the caller interrupts. What you hear is
phone quality, so you can judge the agent the way a caller would. Only the phone network itself
is missing: use a real Twilio number for that (tools/phone_sim/README.md).
"""
import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twilio_side  # noqa: E402

PHONE_RATE = twilio_side.PHONE_RATE
FRAME = twilio_side.FRAME_BYTES


def line_noise(frames: int, rng=np.random.default_rng()) -> bytes:
    """A live line between words: faint hiss, never exact zeros (fusion would take those for a muted mic)."""
    from fusion_runtime.phone import mulaw

    return mulaw.encode(rng.normal(0, 30, frames * FRAME))


def load_speech(path: str) -> bytes:
    """A recording as the caller's voice: mono, 8 kHz, μ-law."""
    import soundfile
    from fusion_runtime.audio.echo_canceller import StreamingResampler
    from fusion_runtime.phone import mulaw

    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    phone = StreamingResampler(rate, PHONE_RATE).process(mono) if rate != PHONE_RATE else mono
    return mulaw.encode(np.clip(phone * 32767, -32768, 32767))


async def speak_recording(call, speech: bytes, listen_s: float, note, heard_at: list) -> None:
    """Wait for the greeting, say the recording, listen for the answer, hang up."""
    await wait_for_quiet(call, first_reply_s=10.0)
    note(f"caller: (playing {len(speech) / PHONE_RATE:.1f} s of recorded speech)")
    started = time.monotonic()
    for i in range(0, len(speech) - FRAME + 1, FRAME):
        await call.send_caller_audio(speech[i:i + FRAME])
        await asyncio.sleep(max(0.0, started + (i // FRAME + 1) * 0.02 - time.monotonic()))
    stopped = time.monotonic()
    heard_before = call.agent_audio_s
    deadline = stopped + listen_s
    while time.monotonic() < deadline and call.closed is None:
        await call.send_caller_audio(line_noise(1))
        await asyncio.sleep(0.02)
        if call.agent_audio_s > heard_before and call.playback.ahead_s() == 0:
            break  # the answer has played in full
    if call.agent_audio_s > heard_before:
        first = next((t for t in heard_at if t > stopped), stopped)
        note(f"agent answered: {call.agent_audio_s - heard_before:.1f} s of audio, the first of it "
             f"{first - stopped:.2f} s after the caller stopped (fusion's turn wait included)")
    else:
        note("agent didn't answer")


async def wait_for_quiet(call, first_reply_s: float) -> None:
    """Until the greeting (if any) has played, sending line noise the way a live line does."""
    deadline = time.monotonic() + first_reply_s
    while time.monotonic() < deadline and call.closed is None:
        await call.send_caller_audio(line_noise(1))
        await asyncio.sleep(0.02)
        if call.agent_audio_s > 0 and call.playback.ahead_s() == 0 and not call.playback.marks:
            return


async def dead_line(call, seconds: float, note) -> None:
    """A call whose audio path never attached: the line's faint hiss and nothing else."""
    note(f"caller: (a dead line: {seconds:.0f} s of hiss, nobody there)")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and call.closed is None:
        await call.send_caller_audio(line_noise(1))
        await asyncio.sleep(0.02)


async def talk(call, audio, note) -> None:
    """The microphone, cleaned of the agent's own voice, as the caller's side of the line."""
    from fusion_runtime.phone import mulaw

    note("connected. Talk; interrupt the agent whenever you like. Ctrl+C hangs up.")
    while call.closed is None:
        chunk = await asyncio.to_thread(audio.read, 0.1)
        if chunk is None:
            continue
        frames = len(chunk.pcm16) // 2 // FRAME
        if chunk.safe_to_send:
            await call.send_caller_audio(mulaw.encode(np.frombuffer(chunk.pcm16, dtype=np.int16)))
        else:
            await call.send_caller_audio(line_noise(frames))  # echo not cancelled yet: send the line, not the room


async def main(args) -> int:
    def note(text):
        print(f"[phone] {text}", flush=True)

    token = args.auth_token or os.environ.get("FUSION_TWILIO_AUTH_TOKEN", "")
    if not token:
        note("set FUSION_TWILIO_AUTH_TOKEN (the same value the server has), or pass --auth-token")
        return 2

    recorded, heard_at = [], []
    audio = None
    if args.say or args.dead_line:
        playback = twilio_side.Playback()
        speech = load_speech(args.say) if args.say else b""
    else:
        from fusion_runtime.audio.duplex_audio import DuplexAudio

        audio = DuplexAudio(output_rate=PHONE_RATE, chunk_ms=20, echo_cancellation=not args.no_aec)
        audio.start()
        playback = twilio_side.Playback(queued_s=lambda: audio.queued_seconds)

    def on_audio(pcm):
        recorded.append(pcm)
        heard_at.append(time.monotonic())
        if audio is not None:
            audio.play(pcm, PHONE_RATE)

    note(f"calling {args.url} as {args.caller}")
    try:
        call = await twilio_side.place_call(
            args.url, token, playback, caller=args.caller, on_audio=on_audio, on_note=note,
            on_clear=(audio.flush if audio is not None else lambda: None))
    except Exception as e:
        note(f"the call didn't connect: {e}")
        if audio is not None:
            audio.stop()
        return 1

    listener = asyncio.create_task(call.run())
    try:
        if args.dead_line:
            await dead_line(call, args.dead_line, note)
        elif args.say:
            await speak_recording(call, speech, args.listen, note, heard_at)
        else:
            await talk(call, audio, note)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if call.closed is None:
            await call.hang_up()
            note("hung up")
        await asyncio.wait_for(listener, 5)
        if audio is not None:
            audio.stop()
    note(f"call ended: {call.closed}. Agent audio {call.agent_audio_s:.1f} s, interrupted {call.clears} time(s)")
    if args.record and recorded:
        with wave.open(args.record, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(PHONE_RATE)
            out.writeframes(b"".join(recorded))
        note(f"the agent's side of the call, at phone quality: {args.record}")
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Call a fusion agent like a Twilio phone call, from this machine.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the fusion server")
    parser.add_argument("--auth-token", help="the Twilio auth token the server checks (default: $FUSION_TWILIO_AUTH_TOKEN)")
    parser.add_argument("--say", metavar="WAV", help="play this recording as the caller instead of the microphone")
    parser.add_argument("--listen", type=float, default=15.0, help="with --say: seconds to wait for the answer")
    parser.add_argument("--record", metavar="WAV", help="save what the agent said, at phone quality")
    parser.add_argument("--caller", default="+15550100001", help="the caller's number, as Twilio would send it")
    parser.add_argument("--no-aec", action="store_true", help="no echo cancellation (use headphones)")
    parser.add_argument("--dead-line", type=float, metavar="S",
                        help="a call whose audio never attached: S seconds of line hiss, then hang up")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse())))
    except KeyboardInterrupt:
        pass
