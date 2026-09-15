"""
Acoustic echo cancellation for the device audio layer.

Why this exists: on laptop speakers the bot's own voice reaches the
microphone about as loud as the user does, so no volume threshold can tell
them apart, and Whisper transcribes the bot's echo as if it were a new user
turn. The fix has to run where both audio streams are known exactly — on the
device, before anything is sent. `fusion_runtime/duplex_audio.py` is the
engine that feeds this.

Written from scratch for fusion-runtime; no code is taken from other voice
frameworks or DSP libraries. The building blocks are standard textbook signal
processing:

* An STFT filter bank (square-root Hann window, 50% overlap) that
  reconstructs its input exactly when nothing is changed — so while the bot
  is silent, the user's voice passes through untouched.
* One short adaptive FIR filter per frequency bin, modelling the
  speaker -> room -> microphone path. It is updated with normalized LMS,
  with the step size taken from a scalar Kalman model of how wrong the
  filter currently is: it adapts quickly while the estimate is poor, and
  slows down by itself while the user talks over the bot (which would
  otherwise corrupt the estimate).
* A residual echo suppressor: a per-bin gain for echo the linear filter
  can't model, such as a small laptop speaker distorting. Kept aggressive,
  because even quiet leftover echo gets transcribed.
* Bulk delay estimation by generalized cross-correlation with phase
  transform (GCC-PHAT), so the device latency doesn't need to be known
  exactly up front.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import gcd, log2
from typing import Optional

import numpy as np
from scipy.signal import firwin

_EPS = 1e-12


def _db_to_power(db: float) -> float:
    return 10.0 ** (db / 10.0)


class StreamingResampler:
    """Rational-ratio resampler that accepts any block size.

    The microphone and the speaker reference must go through identical
    resamplers so they keep the same timing relationship. The canceller
    doesn't care how much delay resampling adds, only that both streams get
    the same amount.
    """

    def __init__(self, rate_in: int, rate_out: int, taps_per_ratio: int = 64):
        if rate_in <= 0 or rate_out <= 0:
            raise ValueError("sample rates must be positive")
        common = gcd(rate_in, rate_out)
        self.up = rate_out // common
        self.down = rate_in // common
        self.rate_in = rate_in
        self.rate_out = rate_out
        if self.up == self.down:
            self._bank = None
            return

        ratio = max(self.up, self.down)
        per_phase = -(-taps_per_ratio * ratio // self.up)  # ceiling division
        # Lowpass at the upsampled rate, a little under the lower Nyquist
        # frequency, sharp enough that almost nothing folds back into the
        # speech band.
        taps = firwin(per_phase * self.up, 0.9 / ratio, window=("kaiser", 8.6)) * self.up
        # Polyphase bank: one row per output phase, one column per input tap.
        self._bank = taps.reshape(per_phase, self.up).T.copy()
        self._per_phase = per_phase
        self._back = np.arange(per_phase)
        self._history = np.zeros(per_phase - 1)
        self._n_in = 0
        self._n_out = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float64).ravel()
        if self._bank is None:
            return block.copy()

        x = np.concatenate([self._history, block])
        n_in = self._n_in + block.size
        # Output j is ready once the newest input it needs has arrived.
        n_out = (n_in * self.up - 1) // self.down + 1 if n_in else 0
        j = np.arange(self._n_out, n_out)
        if j.size:
            position = j * self.down
            newest = position // self.up
            phase = position % self.up
            first = self._n_in - (self._per_phase - 1)  # global index of x[0]
            idx = (newest - first)[:, None] - self._back[None, :]
            out = np.einsum("ij,ij->i", x[idx], self._bank[phase])
        else:
            out = np.zeros(0)

        keep = self._per_phase - 1
        self._history = x[-keep:] if keep else np.zeros(0)
        self._n_in = n_in
        self._n_out = n_out
        return out


class DelayEstimator:
    """Measures how far the microphone lags the speaker reference.

    GCC-PHAT over the last ~1 s: whitening the cross-spectrum keeps only
    phase, so the direct speaker-to-microphone path shows up as one sharp
    peak instead of being smeared by the voice's own spectrum or the room's
    reflections. Estimates are only taken while the bot is actually audible,
    and only confident ones are kept.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_s: float = 1.0,
        update_s: float = 0.25,
        max_delay_s: float = 0.5,
        band_hz: tuple = (200.0, 3800.0),
        min_reference_db: float = -50.0,
        min_peak_ratio: float = 6.0,
        keep: int = 5,
    ):
        self.sample_rate = sample_rate
        self._n = 1 << int(round(log2(sample_rate * window_s)))
        self._max_lag = min(int(max_delay_s * sample_rate), self._n // 2 - 1)
        self._update_every = max(1, int(update_s * sample_rate))
        self._mic = np.zeros(self._n)
        self._ref = np.zeros(self._n)
        self._pos = 0
        self._filled = 0
        self._since_update = 0
        self._window = np.hanning(self._n)
        freqs = np.fft.rfftfreq(self._n, 1.0 / sample_rate)
        self._band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        self._min_ref_power = _db_to_power(min_reference_db)
        self._min_peak_ratio = min_peak_ratio
        self._accepted: deque = deque(maxlen=keep)

    @property
    def delay_samples(self) -> Optional[int]:
        """Median of recent confident estimates, or None until there are enough."""
        if len(self._accepted) < 3:
            return None
        return int(np.median(self._accepted))

    def push(self, mic: np.ndarray, reference: np.ndarray):
        mic = np.asarray(mic, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()
        n = mic.size
        if n > self._n:
            mic, reference, n = mic[-self._n:], reference[-self._n:], self._n
        end = self._pos + n
        if end <= self._n:
            self._mic[self._pos:end] = mic
            self._ref[self._pos:end] = reference
        else:
            split = self._n - self._pos
            self._mic[self._pos:] = mic[:split]
            self._mic[: n - split] = mic[split:]
            self._ref[self._pos:] = reference[:split]
            self._ref[: n - split] = reference[split:]
        self._pos = end % self._n
        self._filled = min(self._n, self._filled + n)
        self._since_update += n
        if self._filled == self._n and self._since_update >= self._update_every:
            self._since_update = 0
            self._estimate()

    def _estimate(self):
        reference = np.roll(self._ref, -self._pos)
        if float(np.mean(reference * reference)) < self._min_ref_power:
            return  # bot not audible enough to measure anything
        mic = np.roll(self._mic, -self._pos)
        cross = np.fft.rfft(mic * self._window) * np.conj(np.fft.rfft(reference * self._window))
        magnitude = np.abs(cross)
        whitened = np.where(self._band, cross / (magnitude + 1e-9 * magnitude.max() + _EPS), 0.0)
        correlation = np.fft.irfft(whitened, n=self._n)[: self._max_lag + 1]
        lag = int(np.argmax(correlation))
        peak = correlation[lag]
        if peak <= 0 or peak / (np.mean(np.abs(correlation)) + _EPS) < self._min_peak_ratio:
            return  # no clear single path (e.g. the user drowning out the bot)
        self._accepted.append(lag)


@dataclass(frozen=True)
class EchoStats:
    """A snapshot of how the canceller is doing, for logging and for
    deciding whether it's safe to leave the microphone open while the bot
    speaks."""

    erle_db: float  # echo removed while bot audio is present (energy-weighted, smoothed)
    delay_ms: float  # speaker -> microphone delay currently assumed
    far_end_active: bool  # bot audio is expected in the current microphone frame
    double_talk: bool  # the microphone doesn't look like pure echo right now
    converged: bool  # has held good cancellation for a while, at least once


class EchoCanceller:
    """Removes the bot's own voice from the microphone signal.

    Feed it equal-length blocks of microphone audio and of the exact audio
    that was sent to the speaker over the same span of time (both at
    `sample_rate`). It returns a block of the same length, delayed by
    `latency_samples`.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        filter_frames: int = 16,
        pre_delay_frames: int = 3,
        max_delay_ms: float = 500.0,
        initial_delay_ms: Optional[float] = None,
        suppression_floor_db: float = -40.0,
        over_subtraction: float = 2.0,
        far_end_floor_db: float = -55.0,
        converged_erle_db: float = 18.0,
        converged_after_s: float = 0.5,
    ):
        if frame_size % 2:
            raise ValueError("frame_size must be even")
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop = frame_size // 2
        # One hop of buffering in `process` plus one hop of overlap-add.
        self.latency_samples = frame_size
        bins = frame_size // 2 + 1

        self._window = np.sqrt(np.hanning(frame_size + 1)[:-1])  # periodic Hann, square-rooted
        self._mic_frame = np.zeros(frame_size)
        self._ref_frame = np.zeros(frame_size)
        self._overlap = np.zeros(self.hop)
        self._pending_mic = np.zeros(0)
        self._pending_ref = np.zeros(0)
        self._output = np.zeros(self.hop)

        # Reference spectra, far enough back to cover the largest delay plus the filter.
        self._taps = filter_frames
        self._pre_delay = pre_delay_frames
        self._max_delay_frames = int(round(max_delay_ms / 1000.0 * sample_rate / self.hop))
        self._history_len = self._max_delay_frames + filter_frames + 1
        self._ref_history = np.zeros((self._history_len, bins), dtype=np.complex128)
        self._tap_offsets = np.arange(filter_frames)
        self._frame_index = 0

        # Linear echo path model and its misalignment (Kalman) state.
        self._weights = np.zeros((bins, filter_frames), dtype=np.complex128)
        self._p_initial = 0.5 / filter_frames
        self._p = np.full(bins, self._p_initial)
        self._p_floor = 1e-5 / filter_frames
        self._path_stability = 0.9998**2  # how slowly we expect the room to change
        self._ref_power = np.zeros(bins)
        self._err_fast = np.zeros(bins)
        # Error/reference correlation, used to tell "the filter is wrong" apart
        # from "the user is talking" (see _process_hop).
        self._corr_smoothing = 0.98
        # Expected size of that correlation from chance alone; doubled because
        # overlapping STFT frames aren't independent samples.
        self._corr_bias = 2.0 * filter_frames * (1 - self._corr_smoothing) / (1 + self._corr_smoothing)
        self._err_ref = np.zeros((bins, filter_frames), dtype=np.complex128)
        self._ref_tap_power = np.zeros((bins, filter_frames))
        self._err_slow = np.zeros(bins)

        # Residual echo suppression state.
        self._s_mic = np.zeros(bins)
        self._s_echo = np.zeros(bins)
        self._s_err = np.zeros(bins)
        self._s_cross = np.zeros(bins, dtype=np.complex128)
        self._leak = np.ones(bins)
        self._residual = np.zeros(bins)
        self._gain = np.ones(bins)
        self._gain_floor_power = _db_to_power(suppression_floor_db)
        self._over_subtraction = over_subtraction
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        self._speech_bins = (freqs >= 300.0) & (freqs <= 3400.0)

        # Rough full-scale spectral energy of a frame (sqrt-Hann window), used
        # to turn dBFS thresholds into this filter bank's units.
        frame_energy_per_unit_power = frame_size * frame_size / 4.0
        self._far_end_floor = filter_frames * frame_energy_per_unit_power * _db_to_power(far_end_floor_db)

        # Delay tracking.
        hint_ms = 80.0 if initial_delay_ms is None else initial_delay_ms
        self._delay_frames = self._frames_for_samples(hint_ms / 1000.0 * sample_rate)
        self._delay_estimator = DelayEstimator(sample_rate, max_delay_s=max_delay_ms / 1000.0)

        # Health metrics.
        self._erle_mic = 0.0
        self._erle_out = 0.0
        self._converged_erle_db = converged_erle_db
        self._converged_frames = int(round(converged_after_s * sample_rate / self.hop))
        self._good_frames = 0
        self._converged = False
        self._far_end_active = False
        self._double_talk = False

    # ------------------------------------------------------------------ API

    @property
    def stats(self) -> EchoStats:
        erle = 10.0 * np.log10((self._erle_mic + _EPS) / (self._erle_out + _EPS)) if self._erle_mic else 0.0
        return EchoStats(
            erle_db=float(erle),
            delay_ms=self._delay_frames * self.hop / self.sample_rate * 1000.0,
            far_end_active=self._far_end_active,
            double_talk=self._double_talk,
            converged=self._converged,
        )

    def process(self, mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()
        if mic.size != reference.size:
            raise ValueError("mic and reference blocks must be the same length")

        self._pending_mic = np.concatenate([self._pending_mic, mic])
        self._pending_ref = np.concatenate([self._pending_ref, reference])
        hop = self.hop
        hops = self._pending_mic.size // hop
        if hops:
            produced = np.empty(hops * hop)
            for i in range(hops):
                span = slice(i * hop, (i + 1) * hop)
                mic_hop, ref_hop = self._pending_mic[span], self._pending_ref[span]
                self._delay_estimator.push(mic_hop, ref_hop)
                self._follow_delay_estimate()
                produced[span] = self._process_hop(mic_hop, ref_hop)
            self._output = np.concatenate([self._output, produced])
            self._pending_mic = self._pending_mic[hops * hop:]
            self._pending_ref = self._pending_ref[hops * hop:]

        out = self._output[: mic.size]
        self._output = self._output[mic.size:]
        return out.astype(np.float32)

    # ------------------------------------------------------------- internals

    def _frames_for_samples(self, samples: float) -> int:
        return int(np.clip(round(samples / self.hop), 0, self._max_delay_frames))

    def _follow_delay_estimate(self):
        estimate = self._delay_estimator.delay_samples
        if estimate is None:
            return
        frames = self._frames_for_samples(estimate)
        change = frames - self._delay_frames
        if abs(change) < 2:
            return  # within what the filter's own taps already cover
        self._shift_filter(change)
        self._delay_frames = frames

    def _shift_filter(self, change: int):
        """Keep the learned echo path when the assumed delay moves, by
        sliding the taps instead of starting over."""
        taps = self._taps
        if abs(change) >= taps:
            self._weights[:] = 0
            self._p[:] = self._p_initial
            self._converged = False
            self._good_frames = 0
            return
        if change > 0:
            self._weights[:, : taps - change] = self._weights[:, change:].copy()
            self._weights[:, taps - change:] = 0
        else:
            shift = -change
            self._weights[:, shift:] = self._weights[:, : taps - shift].copy()
            self._weights[:, :shift] = 0
        self._p = np.maximum(self._p, 0.25 * self._p_initial)

    def _process_hop(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        hop = self.hop
        self._mic_frame[:-hop] = self._mic_frame[hop:]
        self._mic_frame[-hop:] = mic_hop
        self._ref_frame[:-hop] = self._ref_frame[hop:]
        self._ref_frame[-hop:] = ref_hop

        mic_spec = np.fft.rfft(self._mic_frame * self._window)
        ref_spec = np.fft.rfft(self._ref_frame * self._window)

        t = self._frame_index
        self._frame_index += 1
        self._ref_history[t % self._history_len] = ref_spec
        start = max(self._delay_frames - self._pre_delay, 0)
        rows = (t - start - self._tap_offsets) % self._history_len
        ref_taps = self._ref_history[rows].T  # bins x taps

        echo = np.einsum("kl,kl->k", self._weights, ref_taps)
        err = mic_spec - echo
        err_power = err.real**2 + err.imag**2
        ref_energy = (ref_taps.real**2 + ref_taps.imag**2).sum(axis=1)

        # --- linear echo path: NLMS with a Kalman-derived step size --------
        self._ref_power = 0.995 * self._ref_power + 0.005 * (ref_spec.real**2 + ref_spec.imag**2)
        regularized = ref_energy + 1e-3 * self._taps * self._ref_power + _EPS
        self._err_fast = 0.6 * self._err_fast + 0.4 * err_power

        # Error power alone can't tell a wrong filter from the user talking:
        # both just make the error big. What differs is correlation — echo the
        # filter failed to model stays correlated with the reference, the
        # user's voice doesn't. Measure that and raise the misalignment
        # estimate to match, so the filter reopens after the echo path changes
        # (volume, laptop lid, someone moving) instead of freezing forever.
        c = self._corr_smoothing
        self._err_ref = c * self._err_ref + (1 - c) * (err[:, None] * np.conj(ref_taps))
        self._ref_tap_power = c * self._ref_tap_power + (1 - c) * (ref_taps.real**2 + ref_taps.imag**2)
        self._err_slow = c * self._err_slow + (1 - c) * err_power
        explained = ((self._err_ref.real**2 + self._err_ref.imag**2) / (self._ref_tap_power + _EPS)).sum(axis=1)
        unmodelled = np.maximum(explained - self._corr_bias * self._err_slow, 0.0)
        needed = unmodelled / (self._ref_tap_power.sum(axis=1) + _EPS)
        self._p = np.minimum(np.maximum(self._p, needed), 10.0 * self._p_initial)

        misalignment = self._p * ref_energy
        # Whatever error isn't explained by the filter being wrong is the
        # user (or noise) — the more of it there is, the smaller the step.
        near_end = np.maximum(self._err_fast - misalignment, _EPS)
        step = misalignment / (misalignment + near_end)
        self._weights += ((step / regularized) * err)[:, None] * np.conj(ref_taps)
        tap_energy = (self._weights.real**2 + self._weights.imag**2).mean(axis=1)
        self._p = (
            self._path_stability * self._p * (1.0 - step * (ref_energy / regularized) / self._taps)
            + (1.0 - self._path_stability) * tap_energy
            + self._p_floor
        )

        # --- residual echo suppression ------------------------------------
        mic_power = mic_spec.real**2 + mic_spec.imag**2
        echo_power = echo.real**2 + echo.imag**2
        a = 0.85
        self._s_mic = a * self._s_mic + (1 - a) * mic_power
        self._s_echo = a * self._s_echo + (1 - a) * echo_power
        self._s_err = a * self._s_err + (1 - a) * err_power
        self._s_cross = a * self._s_cross + (1 - a) * (mic_spec * np.conj(echo))
        coherence = (self._s_cross.real**2 + self._s_cross.imag**2) / (self._s_mic * self._s_echo + _EPS)

        far_end_active = float(ref_energy.sum()) > self._far_end_floor
        echo_like = far_end_active and float(coherence[self._speech_bins].mean()) > 0.5
        if echo_like:
            # Learn how much echo the linear filter leaves behind, per bin,
            # only while the microphone is clearly just echo.
            leak = np.clip(self._s_err / (self._s_echo + _EPS), 1e-3, 1.0)
            self._leak = 0.95 * self._leak + 0.05 * leak

        residual = self._over_subtraction * self._leak * echo_power
        spread = residual.copy()  # distortion smears echo into neighbouring bins
        spread[1:] = np.maximum(spread[1:], residual[:-1])
        spread[:-1] = np.maximum(spread[:-1], residual[1:])
        residual = np.maximum(spread, 0.75 * self._residual)  # reverb outlasts the filter
        self._residual = residual

        gain = np.sqrt(np.clip(1.0 - residual / (err_power + _EPS), self._gain_floor_power, 1.0))
        # Clamp down on echo immediately; open back up a little more gently.
        rising = gain > self._gain
        gain = np.where(rising, self._gain + 0.5 * (gain - self._gain), gain)
        self._gain = gain
        out_spec = err * gain

        # --- health metrics -------------------------------------------------
        if far_end_active:
            out_power = out_spec.real**2 + out_spec.imag**2
            self._erle_mic = 0.97 * self._erle_mic + 0.03 * float(mic_power[self._speech_bins].sum())
            self._erle_out = 0.97 * self._erle_out + 0.03 * float(out_power[self._speech_bins].sum())
            erle_db = 10.0 * np.log10((self._erle_mic + _EPS) / (self._erle_out + _EPS))
            self._good_frames = self._good_frames + 1 if erle_db >= self._converged_erle_db else 0
            if self._good_frames >= self._converged_frames:
                self._converged = True
        self._far_end_active = far_end_active
        self._double_talk = far_end_active and not echo_like

        frame = np.fft.irfft(out_spec, n=self.frame_size) * self._window
        out = self._overlap + frame[:hop]
        self._overlap = frame[hop:].copy()
        return out
