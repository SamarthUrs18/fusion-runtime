"""
Tests for fusion_runtime/echo_canceller.py against simulated rooms.

Everything here is synthetic and deterministic: a real speech recording (the
hello.wav fixture) plays through a simulated speaker (mild overdrive, like a
small laptop speaker pushed loud), a simulated room (a direct path plus
decaying reflections) and a device delay, then lands in the microphone with
a noise floor. Because the tests build the echo themselves, they know
exactly which part of the microphone signal was echo and which was the
user — so they can measure how much echo was removed and how much of the
user survived, which a live session can't tell us.
"""
import os
import time

import numpy as np
import pytest
import soundfile as sf

from fusion_runtime.audio.echo_canceller import DelayEstimator, EchoCanceller, StreamingResampler

FS = 16000
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "hello.wav")


def speech(seed: int, seconds: float, reverse: bool = False, pitch: float = 1.0) -> np.ndarray:
    """`seconds` of speech-like audio: the fixture repeated with varying
    loudness and pauses. `reverse`/`pitch` make a different-sounding talker
    for the user side of double-talk tests."""
    voice, rate = sf.read(FIXTURE, dtype="float64")
    assert rate == FS
    voice = voice / (np.max(np.abs(voice)) + 1e-12) * 0.5
    if reverse:
        voice = voice[::-1]
    if pitch != 1.0:
        voice = np.interp(np.arange(0, voice.size - 1, pitch), np.arange(voice.size), voice)
    rng = np.random.default_rng(seed)
    parts, total, target = [], 0, int(seconds * FS)
    while total < target:
        part = voice * rng.uniform(0.6, 1.0)
        pause = np.zeros(int(rng.uniform(0.05, 0.25) * FS))
        parts += [part, pause]
        total += part.size + pause.size
    return np.concatenate(parts)[:target]


def room(seed: int, direct: float = 0.5, tail_ms: float = 110.0) -> np.ndarray:
    """Impulse response: a direct path, one early reflection, then a
    decaying reverberant tail."""
    rng = np.random.default_rng(seed)
    n = int(tail_ms / 1000 * FS)
    response = 0.12 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.018 * FS))
    response[0] += direct
    response[int(0.003 * FS)] += 0.3 * direct
    return response


def echo_of(far: np.ndarray, response: np.ndarray, delay_s: float, drive: float = 1.8) -> np.ndarray:
    speaker = np.tanh(drive * far) / drive  # small speaker, slightly overdriven
    wet = np.convolve(speaker, response)[: far.size]
    lag = int(delay_s * FS)
    return np.concatenate([np.zeros(lag), wet])[: far.size]


def noise(seed: int, n: int, db: float = -65.0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n) * 10 ** (db / 20)


def run(canceller: EchoCanceller, mic: np.ndarray, reference: np.ndarray, seed: int = 0) -> np.ndarray:
    """Feed the canceller in irregular blocks, like a real audio callback,
    and line the output back up with the input (undoing its fixed latency)."""
    rng = np.random.default_rng(seed)
    pieces, i = [], 0
    while i < mic.size:
        n = int(rng.integers(160, 481))
        pieces.append(canceller.process(mic[i:i + n], reference[i:i + n]))
        i += n
    out = np.concatenate(pieces).astype(np.float64)
    lag = canceller.latency_samples
    return np.concatenate([out[lag:], np.zeros(lag)])


def db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return 10 * np.log10((np.mean(numerator**2) + 1e-20) / (np.mean(denominator**2) + 1e-20))


def seconds(a: float, b: float) -> slice:
    return slice(int(a * FS), int(b * FS))


class TestEchoCanceller:
    def test_removes_echo_while_only_the_bot_speaks(self):
        far = speech(1, 8.0)
        mic = echo_of(far, room(11), delay_s=0.12) + noise(2, far.size)
        canceller = EchoCanceller(initial_delay_ms=120)
        out = run(canceller, mic, far)
        removed = db_ratio(mic[seconds(4, 8)], out[seconds(4, 8)])
        assert removed >= 25.0, f"only {removed:.1f} dB of echo removed"
        assert canceller.stats.converged

    def test_user_voice_survives_talking_over_the_bot(self):
        far = speech(3, 10.0)
        user = np.zeros_like(far)
        talk = seconds(4.0, 6.5)
        user[talk] = speech(4, 2.5, reverse=True, pitch=0.85)
        mic = echo_of(far, room(12), 0.12) + user + noise(5, far.size)
        canceller = EchoCanceller(initial_delay_ms=120)
        out = run(canceller, mic, far)

        during = seconds(4.2, 6.3)
        kept_db = db_ratio(out[during], user[during])
        correlation = np.corrcoef(out[during], user[during])[0, 1]
        assert kept_db >= -6.0, f"user's voice attenuated by {-kept_db:.1f} dB"
        assert correlation >= 0.7, f"user's voice distorted (correlation {correlation:.2f})"

        after = seconds(7.5, 10.0)
        removed = db_ratio(mic[after], out[after])
        assert removed >= 20.0, f"filter diverged while the user talked ({removed:.1f} dB after)"

    def test_user_passes_through_untouched_while_the_bot_is_silent(self):
        mic = speech(6, 3.0) + noise(7, 3 * FS)
        canceller = EchoCanceller()
        out = run(canceller, mic, np.zeros_like(mic))
        n = mic.size - canceller.latency_samples
        assert np.max(np.abs(out[:n] - mic[:n])) < 1e-6

    def test_finds_the_real_delay_when_the_hint_is_wrong(self):
        far = speech(8, 10.0)
        mic = echo_of(far, room(13), delay_s=0.18) + noise(9, far.size)
        canceller = EchoCanceller(initial_delay_ms=40)
        out = run(canceller, mic, far)
        assert abs(canceller.stats.delay_ms - 180.0) <= 16.0, canceller.stats
        removed = db_ratio(mic[seconds(7, 10)], out[seconds(7, 10)])
        assert removed >= 20.0, f"only {removed:.1f} dB removed after finding the delay"

    def test_recovers_when_the_echo_path_changes(self):
        # e.g. the volume goes up or the laptop lid moves mid-conversation
        far = speech(10, 12.0)
        half = far.size // 2
        mic = np.concatenate([
            echo_of(far, room(14, direct=0.5), 0.12)[:half],
            echo_of(far, room(15, direct=0.9), 0.12)[half:],
        ]) + noise(11, far.size)
        canceller = EchoCanceller(initial_delay_ms=120)
        out = run(canceller, mic, far)
        removed = db_ratio(mic[seconds(9, 12)], out[seconds(9, 12)])
        assert removed >= 20.0, f"only {removed:.1f} dB removed after the path changed"

    def test_runs_comfortably_faster_than_real_time(self):
        far = speech(12, 10.0)
        mic = echo_of(far, room(16), 0.1) + noise(13, far.size)
        canceller = EchoCanceller(initial_delay_ms=100)
        started = time.perf_counter()
        run(canceller, mic, far)
        elapsed = time.perf_counter() - started
        assert elapsed < 2.5, f"10 s of audio took {elapsed:.2f} s"


class TestDelayEstimator:
    def test_locks_onto_the_true_delay(self):
        far = speech(20, 4.0)
        mic = echo_of(far, room(21), 0.2) + noise(22, far.size)
        estimator = DelayEstimator(FS)
        for i in range(0, far.size, 128):
            estimator.push(mic[i:i + 128], far[i:i + 128])
        assert estimator.delay_samples is not None
        assert abs(estimator.delay_samples - int(0.2 * FS)) <= 32

    def test_gives_no_estimate_without_bot_audio(self):
        mic = speech(23, 3.0)  # the user talking, bot silent
        silent = np.zeros_like(mic)
        estimator = DelayEstimator(FS)
        for i in range(0, mic.size, 128):
            estimator.push(mic[i:i + 128], silent[i:i + 128])
        assert estimator.delay_samples is None


class TestStreamingResampler:
    def test_keeps_the_speech_band_and_blocks_aliasing(self):
        t = np.arange(24000) / 24000
        in_band = StreamingResampler(24000, 16000).process(np.sin(2 * np.pi * 1000 * t))
        # 10 kHz can't exist at 16 kHz; unfiltered it would fold to 6 kHz.
        out_of_band = StreamingResampler(24000, 16000).process(np.sin(2 * np.pi * 10000 * t))
        steady = slice(1600, 14400)
        assert abs(20 * np.log10(np.sqrt(2) * np.std(in_band[steady]))) < 0.2
        assert 20 * np.log10(np.sqrt(2) * np.std(out_of_band[steady]) + 1e-12) < -50

    def test_block_size_does_not_change_the_result(self):
        x = np.random.default_rng(30).standard_normal(24000)
        whole = StreamingResampler(24000, 16000).process(x)
        resampler = StreamingResampler(24000, 16000)
        rng = np.random.default_rng(31)
        pieces, i = [], 0
        while i < x.size:
            n = int(rng.integers(1, 700))
            pieces.append(resampler.process(x[i:i + n]))
            i += n
        stitched = np.concatenate(pieces)
        assert stitched.size == whole.size
        assert np.allclose(stitched, whole, atol=1e-12)

    def test_output_length_follows_the_rate_ratio(self):
        resampler = StreamingResampler(48000, 16000)
        total = sum(resampler.process(np.zeros(480)).size for _ in range(100))
        assert total == 16000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
