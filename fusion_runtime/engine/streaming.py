"""Streaming transcription on top of STT runtimes that transcribe whole utterances.

Whisper-style models aren't streaming: they take a clip and return text. To
show partial transcripts and find the end of a turn, the engine re-transcribes
the turn's audio so far every `step_s` seconds and emits each result as a
PartialTranscript. This strategy lives in the engine, not in a runtime, so
every such model gets it and a real streaming model can replace it later.
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
    re-transcriptions is what produced garbled, duplicated transcripts. A
    turn's accumulation is reset through `rolling_transcripts`' reset_signal.
    """

    text: str
    is_final: bool
    confidence: float
    latency_ms: float
    language: Optional[str] = None


Transcribe = Callable[[bytes], Awaitable[Transcript]]


async def rolling_transcripts(
    audio_chunks: AsyncIterator[bytes],
    transcribe: Transcribe,
    reset_signal: Optional[asyncio.Event] = None,
    step_s: float = 1.0,
    max_window_s: float = 30.0,
) -> AsyncIterator[PartialTranscript]:
    """Re-transcribe the current turn every `step_s` seconds of new audio.

    Each emission re-transcribes the whole turn so far, not a sliding
    sub-window. The buffer is already scoped to one turn (cleared on
    reset_signal), so a sub-window bought nothing and cost a lot: every step
    re-transcribed audio the previous step had already covered, so
    consecutive results overlapped heavily and concatenating them produced
    garbled, duplicated transcripts. Transcribing the full turn also gives
    Whisper real context. `max_window_s` (Whisper's context length) caps the
    worst case.

    `reset_signal` is set by the caller once a turn has ended; the buffer is
    cleared on the next chunk so already-transcribed speech isn't re-emitted
    into the next turn.
    """
    max_bytes = int(max_window_s * SAMPLE_RATE) * BYTES_PER_SAMPLE
    step_samples = int(step_s * SAMPLE_RATE)
    audio_buffer = bytearray()
    last_emit_samples = 0

    async def emit(is_final: Optional[bool]) -> Optional[PartialTranscript]:
        window = bytes(audio_buffer[-max_bytes:]) if len(audio_buffer) > max_bytes else bytes(audio_buffer)
        started = time.perf_counter()
        result = await transcribe(window)
        text = result.text
        final = is_final if is_final is not None else bool(text.strip()) and text.strip()[-1] in ".!?。！？"
        return PartialTranscript(
            text=text,
            is_final=final,
            confidence=result.confidence if result.confidence is not None else 0.0,
            latency_ms=(time.perf_counter() - started) * 1000,
            language=result.language,
        )

    async for chunk in audio_chunks:
        if reset_signal is not None and reset_signal.is_set():
            audio_buffer.clear()
            last_emit_samples = 0
            reset_signal.clear()

        audio_buffer.extend(chunk)
        current_samples = len(audio_buffer) // BYTES_PER_SAMPLE
        if current_samples - last_emit_samples >= step_samples and current_samples >= step_samples:
            # Whisper's own punctuation guess; the engine's silence watcher decides real turn ends
            yield await emit(None)
            last_emit_samples = current_samples

    # Final flush: the stream ended (disconnect, or a finite source) with audio
    # that never reached a turn boundary. Skipped if a reset is pending: a turn
    # already consumed this audio and no later chunk came to process the reset
    # (VAD drops trailing silence). Re-transcribing that tail on its own used to
    # surface as a spurious extra turn (a stray "day." after "...today?").
    if reset_signal is not None and reset_signal.is_set():
        audio_buffer.clear()
        reset_signal.clear()
        return
    if len(audio_buffer) > last_emit_samples * BYTES_PER_SAMPLE:
        yield await emit(True)


def raise_if_error(result) -> Transcript:
    """STT runtimes report per-request failures in the result slot; the engine raises them."""
    if isinstance(result, AdapterError):
        raise result
    return result
