"""
Tests for fusion_runtime/duplex_audio.py without touching audio hardware.

The audio callback is driven by hand with fake driver timestamps, and a
simulated room feeds the speaker output back into the microphone the way a
laptop does — so the whole device path (playback buffer, reference capture,
resampling, echo cancellation, the send gate) runs exactly as it would live.
"""
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from scipy.signal import lfilter

from fusion_runtime.audio.duplex_audio import DuplexAudio
from fusion_runtime.audio.echo_canceller import EchoCanceller, StreamingResampler
from test_echo_canceller import db_ratio, speech

DEVICE_RATE = 24000
CHUNK_S = 0.02


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def tone(seconds: float, freq: float = 440.0, rate: int = DEVICE_RATE) -> np.ndarray:
    return 0.3 * np.sin(2 * np.pi * freq * np.arange(int(seconds * rate)) / rate)


class SimulatedRoom:
    """Speaker output back into the microphone: a delay (device buffers plus
    the air gap) and a short reverberant response."""

    def __init__(self, delay_s: float, gain: float = 0.5, seed: int = 0, rate: int = DEVICE_RATE):
        rng = np.random.default_rng(seed)
        n = int(0.03 * rate)
        self._response = 0.1 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.006 * rate))
        self._response[0] += gain
        self._response[int(0.002 * rate)] += 0.3 * gain
        self._state = np.zeros(n - 1)
        self._delay = np.zeros(int(delay_s * rate))

    def microphone(self, speaker: np.ndarray) -> np.ndarray:
        joined = np.concatenate([self._delay, speaker])
        delayed, self._delay = joined[: speaker.size], joined[speaker.size:]
        wet, self._state = lfilter(self._response, [1.0], delayed, zi=self._state)
        return wet


def run_callbacks(
    audio: DuplexAudio,
    blocks: int,
    room: Optional[SimulatedRoom] = None,
    user: Optional[np.ndarray] = None,
    noise_db: float = -65.0,
    start: float = 100.0,
):
    """Drive the audio callback `blocks` times, like the driver would.
    Returns (raw microphone signal, everything sent to the speaker)."""
    frames = audio.block_frames
    rng = np.random.default_rng(7)
    previous_speaker = np.zeros(frames)
    mic_all, speaker_all = [], []
    for k in range(blocks):
        mic = rng.standard_normal(frames) * 10 ** (noise_db / 20)
        if user is not None:
            part = user[k * frames:(k + 1) * frames]
            mic[: part.size] += part
        if room is not None:
            mic = mic + room.microphone(previous_speaker)
        indata = mic.astype(np.float32)[:, None]
        outdata = np.zeros((frames, 1), dtype=np.float32)
        now = start + k * frames / audio.device_rate
        timing = SimpleNamespace(
            currentTime=now, inputBufferAdcTime=now - 0.008, outputBufferDacTime=now + 0.012
        )
        audio._callback(indata, outdata, frames, timing, None)
        previous_speaker = outdata[:, 0].astype(np.float64)
        mic_all.append(mic)
        speaker_all.append(previous_speaker)
    return np.concatenate(mic_all), np.concatenate(speaker_all)


def read_all(audio: DuplexAudio):
    audio.process_pending()
    chunks = []
    while (chunk := audio.read(timeout=0)) is not None:
        chunks.append(chunk)
    return chunks


def to_float(chunks) -> np.ndarray:
    return np.concatenate([np.frombuffer(c.pcm16, dtype=np.int16) for c in chunks]).astype(np.float64) / 32768.0


class TestPlayback:
    def test_plays_in_order_then_pads_with_silence(self):
        audio = DuplexAudio()
        ramp = np.linspace(-0.5, 0.5, 300)
        audio.play(pcm16(ramp), DEVICE_RATE)
        _, speaker = run_callbacks(audio, blocks=2)
        expected = np.frombuffer(pcm16(ramp), dtype=np.int16) / 32768.0
        assert np.allclose(speaker[:300], expected, atol=1e-6)
        assert not np.any(speaker[300:])

    def test_flush_silences_the_very_next_block(self):
        # The old client wrote whole sentences with a blocking write, so an
        # interruption only took effect once the sentence finished (measured
        # at 1.58 s for 1.5 s of audio).
        audio = DuplexAudio()
        audio.play(pcm16(tone(1.0)), DEVICE_RATE)
        _, before = run_callbacks(audio, blocks=1)
        assert np.any(before)
        audio.flush()
        assert audio.queued_seconds == 0
        _, after = run_callbacks(audio, blocks=1)
        assert not np.any(after)

    def test_resamples_other_rates_to_the_device_rate(self):
        audio = DuplexAudio(device_rate=24000)
        audio.play(pcm16(tone(1.0, rate=16000)), 16000)
        assert audio.queued_seconds == pytest.approx(1.0, abs=0.01)


class TestSendGate:
    def test_holds_back_audio_while_uncancelled_echo_is_possible(self):
        audio = DuplexAudio(echo_cancellation=False)
        audio.play(pcm16(tone(0.2)), DEVICE_RATE)
        run_callbacks(audio, blocks=100)  # 1 s
        chunks = read_all(audio)
        during = chunks[int(0.05 / CHUNK_S):int(0.2 / CHUNK_S)]
        after = chunks[int(0.7 / CHUNK_S):]
        assert during and not any(c.safe_to_send for c in during)
        assert after and all(c.safe_to_send for c in after)


class TestEndToEnd:
    def test_removes_its_own_echo_through_the_whole_device_path(self):
        audio = DuplexAudio()
        voice = StreamingResampler(16000, DEVICE_RATE).process(speech(40, 8.0))
        audio.play(pcm16(voice), DEVICE_RATE)
        mic, _ = run_callbacks(audio, blocks=800, room=SimulatedRoom(delay_s=0.03))
        chunks = read_all(audio)

        cleaned = to_float(chunks)
        raw = StreamingResampler(DEVICE_RATE, 16000).process(mic)
        window = slice(5 * 16000, int(7.5 * 16000))
        removed = db_ratio(raw[window], cleaned[window])
        assert removed >= 25.0, f"only {removed:.1f} dB of echo removed end to end"
        assert audio.echo_stats.converged
        early = chunks[int(0.1 / CHUNK_S):int(0.4 / CHUNK_S)]
        assert not any(c.safe_to_send for c in early), "would have sent echo before cancellation converged"
        assert chunks[-1].safe_to_send

    def test_user_voice_passes_through_while_the_bot_is_silent(self):
        audio = DuplexAudio()
        user = StreamingResampler(16000, DEVICE_RATE).process(speech(41, 2.0))
        mic, _ = run_callbacks(audio, blocks=200, user=user)
        chunks = read_all(audio)
        assert all(c.safe_to_send for c in chunks)

        cleaned = to_float(chunks)
        raw = StreamingResampler(DEVICE_RATE, 16000).process(mic)
        lag = EchoCanceller().latency_samples
        n = min(cleaned.size - lag, raw.size)
        assert np.max(np.abs(cleaned[lag:lag + n] - raw[:n])) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
