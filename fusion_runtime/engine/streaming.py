"""Streaming transcription on top of STT runtimes that transcribe whole utterances.

Whisper-style models aren't streaming: they take a clip and return text. To
show partial transcripts and find the end of a turn, the engine re-transcribes
the turn's audio so far every `step_s` seconds of new speech. When the user
pauses, whatever was said since the last partial is transcribed too
(`TurnTranscriber.finalize`), so a turn never ends on a stale transcript that
is missing its last words. This strategy lives in the engine, not in a
runtime, so every such model gets it and a real streaming model can replace
it later.
"""
import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

from fusion_runtime.contract import AdapterError, Transcript

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass
class PartialTranscript:
    """A streaming transcription result.

    `text` is **cumulative for the current turn**: each result restates the
    whole turn transcribed so far, superseding the previous one, rather than
    carrying only newly recognized words. Consumers must *replace* their
    running transcript with it, not append; appending overlapping
    re-transcriptions is what produced garbled, duplicated transcripts.
    Every result is partial: whether the turn is over is decided by turn
    detection, never by the transcript's punctuation.
    `audio_bytes` is how much of the turn's audio it covers.
    """

    text: str
    confidence: float
    latency_ms: float
    language: Optional[str] = None
    audio_bytes: int = 0


Transcribe = Callable[[bytes], Awaitable[Transcript]]


class TurnTranscriber:
    """The current turn's speech, and its transcripts.

    Each emission re-transcribes the whole turn so far, not a sliding
    sub-window: a sub-window re-transcribed audio the previous step had
    already covered, and stitching overlapping results produced garbled,
    duplicated transcripts. Transcribing the full turn also gives Whisper real
    context. `max_window_s` (Whisper's context length) caps the worst case.

    `reset()` starts a new turn. A transcription that was running when the
    turn was reset belongs to the old turn and is dropped, so it can't leak
    into the next one.
    """

    def __init__(self, transcribe: Transcribe, step_s: float = 1.0, max_window_s: float = 30.0,
                 min_final_s: float = 0.15):
        self._transcribe = transcribe
        self.step_bytes = int(step_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.max_bytes = int(max_window_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.min_final_bytes = int(min_final_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
        self.buffer = bytearray()
        self.transcribed_bytes = 0  # how much of the buffer the latest transcript covers
        self.epoch = 0

    @property
    def untranscribed_bytes(self) -> int:
        return len(self.buffer) - self.transcribed_bytes

    def reset(self) -> None:
        self.buffer.clear()
        self.transcribed_bytes = 0
        self.epoch += 1

    async def feed(self, chunk: bytes) -> Optional[PartialTranscript]:
        """Add speech; transcribe the turn again once `step_s` of new speech has arrived."""
        self.buffer.extend(chunk)
        if self.untranscribed_bytes >= self.step_bytes:
            return await self._run()
        return None

    async def finalize(self) -> Optional[PartialTranscript]:
        """Transcribe the words said since the last partial, if there are enough to matter."""
        if self.untranscribed_bytes < self.min_final_bytes:
            return None
        return await self._run()

    async def _run(self) -> Optional[PartialTranscript]:
        epoch, covered = self.epoch, len(self.buffer)
        window = bytes(self.buffer[-self.max_bytes:]) if covered > self.max_bytes else bytes(self.buffer)
        started = time.perf_counter()
        result = await self._transcribe(window)
        if epoch != self.epoch:
            return None  # the turn ended while this ran: its words belong to that turn
        self.transcribed_bytes = max(self.transcribed_bytes, covered)
        text = result.text
        return PartialTranscript(
            text=text,
            confidence=result.confidence if result.confidence is not None else 0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            language=result.language,
            audio_bytes=covered,
        )

    async def stream(self, audio_chunks: AsyncIterator[bytes],
                     reset_signal: Optional[asyncio.Event] = None) -> AsyncIterator[PartialTranscript]:
        """Partial transcripts as speech arrives, and a final one when the audio ends.

        `reset_signal`, when set by the caller, starts a new turn on the next chunk.
        """
        async for chunk in audio_chunks:
            if reset_signal is not None and reset_signal.is_set():
                self.reset()
                reset_signal.clear()
            result = await self.feed(chunk)
            if result is not None:
                yield result

        # The stream ended (disconnect, or a finite source) with audio that never
        # reached a turn boundary. Skipped if a reset is pending: a turn already
        # consumed this audio and no later chunk came to process the reset (VAD
        # drops trailing silence). Re-transcribing that tail on its own used to
        # surface as a spurious extra turn (a stray "day." after "...today?").
        if reset_signal is not None and reset_signal.is_set():
            self.reset()
            reset_signal.clear()
            return
        if self.untranscribed_bytes > 0:
            result = await self._run()
            if result is not None:
                yield result


async def rolling_transcripts(
    audio_chunks: AsyncIterator[bytes],
    transcribe: Transcribe,
    reset_signal: Optional[asyncio.Event] = None,
    step_s: float = 1.0,
    max_window_s: float = 30.0,
) -> AsyncIterator[PartialTranscript]:
    """Partial transcripts for a stream of speech (see TurnTranscriber)."""
    transcriber = TurnTranscriber(transcribe, step_s=step_s, max_window_s=max_window_s)
    async for result in transcriber.stream(audio_chunks, reset_signal):
        yield result


def raise_if_error(result) -> Transcript:
    """STT runtimes report per-request failures in the result slot; the engine raises them."""
    if isinstance(result, AdapterError):
        raise result
    return result
