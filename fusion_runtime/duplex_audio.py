"""
Full-duplex device audio with echo cancellation, for voice clients.

The microphone and the speaker run in ONE audio callback, so each block sent
to the speaker and the microphone block captured alongside it share the
audio driver's clock. That is what makes echo cancellation work in a plain
Python client: the canceller always gets the exact samples that were played,
and how far apart the two streams are, instead of guessing from a separate
playback loop.

It also makes stopping instant. The callback pulls playback a few
milliseconds at a time, so flush() silences the bot within one block rather
than after the sentence currently playing.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fusion_runtime.echo_canceller import EchoCanceller, EchoStats, StreamingResampler

# After the last audible sample leaves the speaker, the room and the input
# buffers can still hold some of it for a moment.
ECHO_TAIL_S = 0.35


class PlaybackBuffer:
    """Thread-safe queue of speaker samples, pulled one block at a time by
    the audio callback."""

    def __init__(self):
        self._chunks: deque = deque()
        self._head = 0  # samples already taken from _chunks[0]
        self._available = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        return self._available

    def push(self, samples: np.ndarray):
        if samples.size == 0:
            return
        with self._lock:
            self._chunks.append(np.asarray(samples, dtype=np.float32))
            self._available += samples.size

    def pull(self, frames: int) -> np.ndarray:
        """The next `frames` samples, padded with silence if there aren't enough."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            filled = 0
            while filled < frames and self._chunks:
                head = self._chunks[0]
                take = min(frames - filled, head.size - self._head)
                out[filled:filled + take] = head[self._head:self._head + take]
                filled += take
                self._head += take
                if self._head == head.size:
                    self._chunks.popleft()
                    self._head = 0
            self._available -= filled
        return out

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._head = 0
            self._available = 0


@dataclass(frozen=True)
class MicChunk:
    """One chunk of microphone audio, cleaned and ready to send."""

    pcm16: bytes  # mono int16 at DuplexAudio.output_rate
    echo_possible: bool  # the bot's audio could have reached the mic during this chunk
    echo_cancelled: bool  # echo cancellation is running and has converged

    @property
    def safe_to_send(self) -> bool:
        """False only when echo may be in the audio and nothing has removed it."""
        return self.echo_cancelled or not self.echo_possible


class DuplexAudio:
    """Speaker and microphone as one stream, with the bot's own voice removed
    from the microphone before anything leaves the device.

        audio = DuplexAudio()
        audio.start()
        audio.play(tts_pcm16, 24000)   # bot speech
        chunk = audio.read()           # cleaned microphone audio
        if chunk and chunk.safe_to_send:
            send(chunk.pcm16)
        audio.flush()                  # user interrupted: stop right now
    """

    def __init__(
        self,
        device_rate: int = 24000,
        output_rate: int = 16000,
        block_ms: float = 10.0,
        chunk_ms: float = 20.0,
        echo_cancellation: bool = True,
        input_device=None,
        output_device=None,
    ):
        self.device_rate = device_rate
        self.output_rate = output_rate
        self.block_frames = int(device_rate * block_ms / 1000)
        self.echo_cancellation = echo_cancellation
        self._devices = (input_device, output_device)
        self._chunk_samples = int(output_rate * chunk_ms / 1000)

        self._playback = PlaybackBuffer()
        self._playback_resamplers: dict = {}
        self._playback_lock = threading.Lock()
        # Identical resamplers, so the mic and the reference stay aligned.
        self._mic_resampler = StreamingResampler(device_rate, output_rate)
        self._ref_resampler = StreamingResampler(device_rate, output_rate)
        self._canceller: Optional[EchoCanceller] = None

        self._captured: "queue.SimpleQueue" = queue.SimpleQueue()
        self._chunks: "queue.SimpleQueue[MicChunk]" = queue.SimpleQueue()
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_echo = False

        self._last_sound_end = -np.inf  # stream time the last audible sample leaves the speaker
        self._audible_until = 0.0  # the same moment, on time.monotonic()
        self._block_clock = 0.0
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.dropouts = 0  # audio callbacks the driver reported as under/overflowing
        # Optional diagnostics tap, called as monitor(mic, reference, cleaned)
        # with float arrays at output_rate, on the processing thread.
        self.monitor = None

    # ---------------------------------------------------------------- control

    def start(self):
        import sounddevice as sd

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, name="duplex-audio", daemon=True)
        self._thread.start()
        self._stream = sd.Stream(
            samplerate=self.device_rate,
            blocksize=self.block_frames,
            channels=1,
            dtype="float32",
            latency="low",
            device=self._devices,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # --------------------------------------------------------------- playback

    def play(self, pcm16: bytes, sample_rate: int):
        """Queue mono int16 audio for the speaker."""
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != self.device_rate:
            with self._playback_lock:
                resampler = self._playback_resamplers.get(sample_rate)
                if resampler is None:
                    resampler = StreamingResampler(sample_rate, self.device_rate)
                    self._playback_resamplers[sample_rate] = resampler
                samples = resampler.process(samples).astype(np.float32)
        self._playback.push(samples)

    def flush(self):
        """Drop everything queued; the speaker goes quiet within one block."""
        self._playback.clear()
        with self._playback_lock:
            self._playback_resamplers.clear()  # don't carry the old sentence into the next one

    @property
    def queued_seconds(self) -> float:
        return self._playback.available / self.device_rate

    @property
    def bot_audible(self) -> bool:
        """Bot audio is queued or still coming out of the speaker."""
        return self._playback.available > 0 or time.monotonic() < self._audible_until

    # ---------------------------------------------------------------- capture

    def read(self, timeout: float = 0.1) -> Optional[MicChunk]:
        """The next cleaned microphone chunk, or None if none arrived in time."""
        try:
            if timeout <= 0:
                return self._chunks.get_nowait()
            return self._chunks.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def echo_stats(self) -> Optional[EchoStats]:
        canceller = self._canceller
        return canceller.stats if canceller is not None else None

    def process_pending(self):
        """Process everything captured so far on the calling thread. The
        background thread does this during normal use; tests call it directly."""
        while True:
            try:
                item = self._captured.get_nowait()
            except queue.Empty:
                return
            self._process(item)

    # ------------------------------------------------------------- internals

    def _callback(self, indata, outdata, frames, time_info, status):
        # Runs on the audio driver's thread: keep it to copying and bookkeeping.
        if status:
            self.dropouts += 1
        speaker = self._playback.pull(frames)
        outdata[:, 0] = speaker
        now, adc, dac = self._timestamps(time_info, frames)
        if np.any(speaker):
            self._last_sound_end = dac + frames / self.device_rate
            self._audible_until = time.monotonic() + max(0.0, self._last_sound_end - now)
        echo_possible = adc < self._last_sound_end + ECHO_TAIL_S
        # `speaker` is exactly what the driver will play: it's the reference.
        self._captured.put((indata[:, 0].copy(), speaker, echo_possible, dac - adc))

    def _timestamps(self, time_info, frames: int):
        """(now, capture time of this mic block, play time of this speaker
        block), all on the stream's clock."""
        now = float(getattr(time_info, "currentTime", 0.0) or 0.0)
        adc = float(getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0)
        dac = float(getattr(time_info, "outputBufferDacTime", 0.0) or 0.0)
        if adc > 0 and dac > 0:
            return now, adc, dac
        # Some host APIs don't report timestamps: count blocks and use the
        # latency the stream reports instead. The canceller's own delay
        # estimate corrects whatever this gets wrong.
        in_latency, out_latency = (0.01, 0.01)
        if self._stream is not None:
            in_latency, out_latency = self._stream.latency
        clock = self._block_clock
        self._block_clock += frames / self.device_rate
        return clock, clock - in_latency, clock + out_latency

    def _process_loop(self):
        while self._running:
            try:
                item = self._captured.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process(item)

    def _process(self, item):
        mic, speaker, echo_possible, delay_s = item
        mic = self._mic_resampler.process(mic)
        reference = self._ref_resampler.process(speaker)
        if self.echo_cancellation:
            if self._canceller is None:
                self._canceller = EchoCanceller(
                    sample_rate=self.output_rate,
                    initial_delay_ms=max(0.0, delay_s) * 1000.0,
                )
            clean = self._canceller.process(mic, reference)
            cancelled = self._canceller.stats.converged
        else:
            clean = mic.astype(np.float32)
            cancelled = False
        if self.monitor is not None:
            self.monitor(mic, reference, clean)

        self._pending = np.concatenate([self._pending, clean])
        self._pending_echo = self._pending_echo or echo_possible
        while self._pending.size >= self._chunk_samples:
            piece = self._pending[: self._chunk_samples]
            self._pending = self._pending[self._chunk_samples:]
            pcm16 = (np.clip(piece, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            self._chunks.put(MicChunk(pcm16, self._pending_echo, cancelled))
            self._pending_echo = echo_possible
