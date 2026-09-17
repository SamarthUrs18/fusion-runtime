#!/usr/bin/env python3
"""
Measure how much of the bot's own voice the device echo canceller removes on
THIS machine's speakers and microphone. Worth running once on a new machine,
before a live session.

It plays about ten seconds of synthesized speech out loud while recording the
microphone. Stay quiet while it runs.

    python3 scripts/measure_echo.py

It reports the room's noise floor, how loud the bot's voice reaches the
microphone, how much is left after cancellation, and saves WAV files (raw
mic, speaker reference, mic after cancellation) to listen to.
"""
import argparse
import asyncio
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from fusion_runtime.audio.duplex_audio import DuplexAudio  # noqa: E402
from fusion_runtime.audio.echo_canceller import StreamingResampler  # noqa: E402

RATE = 16000  # DuplexAudio's output rate
SPEECH_RATE = 24000
TEXT = (
    "Hi there! I'm checking how well echo cancellation works on these speakers. "
    "This sentence is a little longer, so the canceller has time to learn the room. "
    "Thanks for staying quiet, this only takes a few more seconds."
)


def dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x)) + 1e-12))


def load_speech():
    """Speech at 24 kHz: Kokoro if its models are installed, otherwise the
    test fixture repeated."""
    try:
        from fusion_runtime.config import DEVELOPMENT_CONFIG
        from fusion_runtime.contract import TTSRequest
        from fusion_runtime.registry import create_runtime
        from fusion_runtime.resolver import resolve_stage_config

        async def synthesize():
            tts = create_runtime(resolve_stage_config("tts", DEVELOPMENT_CONFIG.tts).spec)
            await tts.load()
            return b"".join([chunk.pcm async for chunk in tts.synthesize(TTSRequest(text=TEXT, voice=DEVELOPMENT_CONFIG.tts.voice))])

        result = asyncio.run(synthesize())
        return np.frombuffer(result, dtype=np.int16).astype(np.float64) / 32768.0, "Kokoro"
    except Exception as exc:
        voice, rate = sf.read(os.path.join(ROOT, "tests", "fixtures", "hello.wav"), dtype="float64")
        looped = np.tile(np.concatenate([voice, np.zeros(rate // 4)]), 5)
        speech = StreamingResampler(rate, SPEECH_RATE).process(looped)
        return speech, f"tests/fixtures/hello.wav (Kokoro unavailable: {exc})"


def device(value):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-device", help="microphone (index or part of its name); system default if omitted")
    parser.add_argument("--output-device", help="speaker (index or part of its name); system default if omitted")
    args = parser.parse_args()
    os.chdir(ROOT)  # model paths in the config are relative to the repo

    speech, source = load_speech()
    print(f"Speech: {speech.size / SPEECH_RATE:.1f} s from {source}")

    raw, reference, cleaned = [], [], []
    audio = DuplexAudio(input_device=device(args.input_device), output_device=device(args.output_device))
    audio.monitor = lambda mic, ref, clean: (raw.append(mic), reference.append(ref), cleaned.append(clean))

    with audio:
        print("Listening to the room — stay quiet...")
        time.sleep(1.5)
        quiet_end = sum(block.size for block in list(raw))
        print("Playing speech through the speakers — stay quiet...")
        audio.play((np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16).tobytes(), SPEECH_RATE)
        time.sleep(speech.size / SPEECH_RATE + 1.0)
        stats = audio.echo_stats
        dropouts = audio.dropouts

    raw, reference, cleaned = (np.concatenate(list(parts)) for parts in (raw, reference, cleaned))
    n = min(raw.size, reference.size, cleaned.size)
    quiet = slice(int(0.3 * RATE), max(int(0.4 * RATE), quiet_end - RATE // 10))
    settled = quiet_end + 2 * RATE  # give the canceller two seconds to learn the room
    window = RATE // 10
    # Only compare stretches where the bot is actually talking.
    talking = [
        slice(start, start + window)
        for start in range(settled, n - window, window)
        if dbfs(reference[start:start + window]) > -45.0
    ]
    if not talking:
        print("\n❌ No bot speech to measure — playback may have failed. Try again.")
        return 1

    floor = dbfs(raw[quiet])
    echo_db = dbfs(np.concatenate([raw[w] for w in talking]))
    left_db = dbfs(np.concatenate([cleaned[w] for w in talking]))

    print()
    print(f"Room noise floor:        {floor:6.1f} dBFS")
    print(f"Bot voice at the mic:    {echo_db:6.1f} dBFS  ({echo_db - floor:4.1f} dB above the room)")
    print(f"Left after cancellation: {left_db:6.1f} dBFS  ({left_db - floor:4.1f} dB above the room)")
    print(f"Echo removed:            {echo_db - left_db:6.1f} dB")
    if stats is not None:
        print(
            f"Canceller: delay {stats.delay_ms:.0f} ms, "
            f"converged {'yes' if stats.converged else 'no'}, audio dropouts {dropouts}"
        )

    if echo_db - floor < 10:
        print("\n⚠️  The speech barely reached the mic — turn the speaker volume up and run again.")
    elif left_db - floor <= 6:
        print("\n✅ The bot's voice is cancelled down to the room's own noise. Talking over it should work.")
    elif echo_db - left_db >= 20:
        print(
            "\n⚠️  Most of the echo is removed, but some is left above the room noise. Try the speaker a "
            "little quieter; the server's 🪞 self-echo check catches what gets through."
        )
    else:
        print("\n❌ Echo cancellation isn't removing enough on this hardware. Use headphones, or run the client with FUSION_AEC=0.")

    out_dir = tempfile.mkdtemp(prefix="fusion-echo-")
    for name, data in (("1_mic_raw", raw), ("2_speaker_reference", reference), ("3_mic_after_cancellation", cleaned)):
        sf.write(os.path.join(out_dir, f"{name}.wav"), np.clip(data, -1.0, 1.0), RATE, subtype="PCM_16")
    print(f"\nRecordings saved in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
