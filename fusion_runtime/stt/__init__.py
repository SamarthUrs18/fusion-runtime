"""
STT Base Classes and Implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional
import asyncio


@dataclass
class STTResult:
    """A streaming transcription result.

    `text` is **cumulative for the current turn**: each result restates the
    whole turn transcribed so far, superseding the previous one, rather than
    carrying only newly-recognized words. Consumers must therefore *replace*
    their running transcript with it, not append — appending overlapping
    re-transcriptions is what produced garbled, duplicated transcripts.
    A turn's accumulation is reset via `transcribe_stream`'s `reset_signal`.
    """
    text: str
    is_final: bool
    confidence: float
    latency_ms: float
    language: Optional[str] = None


class STTBase(ABC):
    """Base class for all STT providers."""
    
    def __init__(self, config):
        self.config = config
        self._warm = False
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        budget_ms: Optional[int] = None,
        reset_signal: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[STTResult]:
        """Stream transcription with partial results.

        `reset_signal`, when provided, is set by the caller once a turn has
        ended — implementations using a rolling/sliding audio window should
        clear it on the next chunk so old, already-transcribed speech isn't
        re-emitted into the next turn.
        """
        pass
    
    @abstractmethod
    async def transcribe_file(self, audio_path: str) -> STTResult:
        """Transcribe complete audio file."""
        pass
    
    async def warmup(self):
        """Pre-load models, warm up GPU."""
        if not self._warm:
            await self._warmup_impl()
            self._warm = True
    
    @abstractmethod
    async def _warmup_impl(self):
        pass


class FasterWhisperSTT(STTBase):
    """faster-whisper implementation (CTranslate2 backend)."""
    
    @staticmethod
    def _cuda_available() -> bool:
        """Check for CUDA without importing torch (CTranslate2 handles it)."""
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False
    
    async def _warmup_impl(self):
        from faster_whisper import WhisperModel
        
        # Resolve device/compute_type ("auto" adapts to the machine)
        device = self.config.device
        compute_type = self.config.compute_type
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        self.model = WhisperModel(
            self.config.model,
            device=device,
            compute_type=compute_type,
        )
        # Warmup with dummy audio (BytesIO — faster-whisper's VAD wants file-like input)
        import io
        import numpy as np
        import soundfile as sf
        dummy = np.zeros(16000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, dummy, 16000, format="WAV")
        buf.seek(0)
        segments, _ = self.model.transcribe(buf, language=self.config.language)
        list(segments)  # Consume
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        budget_ms: Optional[int] = None,
        reset_signal: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[STTResult]:
        import numpy as np
        import time

        # Each emission re-transcribes the whole turn so far, not a sliding
        # sub-window. The buffer is already scoped to one turn (cleared on
        # reset_signal), so a sub-window bought nothing and cost a lot: every
        # step re-transcribed audio the previous step had already covered, so
        # consecutive results overlapped heavily and concatenating them
        # produced garbled, duplicated transcripts ("but I cannot. but I
        # can't speak to you right. but I can't speak to you right now...").
        # Transcribing the full turn also gives Whisper real context instead
        # of 5-second fragments. `max_duration` only caps the worst case.
        sample_rate = 16000
        max_duration = 30.0    # seconds — Whisper's own context length
        step_duration = 1.0    # seconds (how often to emit partials)
        max_samples = int(max_duration * sample_rate)
        step_samples = int(step_duration * sample_rate)
        bytes_per_sample = 2  # 16-bit

        audio_buffer = bytearray()
        last_emit_samples = 0

        async for chunk in audio_chunks:
            if reset_signal is not None and reset_signal.is_set():
                # A turn just ended — drop everything transcribed so far so
                # the next window doesn't re-include (and re-emit) the
                # previous turn's speech.
                audio_buffer.clear()
                last_emit_samples = 0
                reset_signal.clear()

            audio_buffer.extend(chunk)

            # Check if we have enough for a window
            current_samples = len(audio_buffer) // bytes_per_sample
            
            if current_samples - last_emit_samples >= step_samples and current_samples >= step_samples:
                # Run transcription on current window
                start = time.perf_counter()
                
                # The whole turn so far (capped at max_duration)
                start_idx = max(0, current_samples - max_samples) * bytes_per_sample
                window_audio = bytes(audio_buffer[start_idx:])
                audio_np = np.frombuffer(window_audio, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_np,
                        language=self.config.language,
                        beam_size=self.config.beam_size,
                        vad_filter=self.config.vad_filter,
                        condition_on_previous_text=False,  # Critical for streaming
                    )
                )
                
                text = " ".join([s.text for s in segments])
                latency = (time.perf_counter() - start) * 1000
                
                # Determine if this looks like a final result
                # (In practice, you'd use VAD or EoT to decide)
                is_final = len(text.strip()) > 0 and text.strip()[-1] in ".!?。！？"
                
                yield STTResult(
                    text=text,
                    is_final=is_final,
                    confidence=info.language_probability,
                    latency_ms=latency,
                    language=info.language,
                )
                
                last_emit_samples = current_samples
        
        # Final flush — the stream ended (disconnect, or a finite source)
        # with audio that never reached a turn boundary. Skipped entirely if
        # a reset is pending: that means a turn already consumed this audio
        # and we simply never saw another chunk to process the reset on
        # (VAD drops trailing silence, so nothing arrives after the last
        # word). Without this check the leftover tail gets re-transcribed in
        # isolation and surfaces as a spurious extra turn — a stray "day."
        # after a perfectly good "...how are you doing today?".
        if reset_signal is not None and reset_signal.is_set():
            audio_buffer.clear()
            reset_signal.clear()
            return

        if len(audio_buffer) > last_emit_samples * bytes_per_sample:
            start = time.perf_counter()
            # Cumulative, like every other emission (see STTResult): the
            # whole turn, not just the bit since the last one.
            current_samples = len(audio_buffer) // bytes_per_sample
            start_idx = max(0, current_samples - max_samples) * bytes_per_sample
            remaining_audio = bytes(audio_buffer[start_idx:])
            audio_np = np.frombuffer(remaining_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio_np,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                )
            )
            
            text = " ".join([s.text for s in segments])
            latency = (time.perf_counter() - start) * 1000
            
            yield STTResult(
                text=text,
                is_final=True,
                confidence=info.language_probability,
                latency_ms=latency,
                language=info.language,
            )
    
    async def transcribe_file(self, audio_path: str) -> STTResult:
        """Transcribe complete audio file."""
        import time
        import numpy as np
        
        start = time.perf_counter()
        
        # Read audio file
        import wave
        with wave.open(audio_path, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            sample_rate = wav.getframerate()
        
        # Convert to numpy
        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run transcription
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_np,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
            )
        )
        
        text = " ".join([s.text for s in segments])
        latency = (time.perf_counter() - start) * 1000
        
        return STTResult(
            text=text,
            is_final=True,
            confidence=info.language_probability,
            latency_ms=latency,
            language=info.language,
        )


class DeepgramSTT(STTBase):
    """Deepgram API implementation."""
    
    async def _warmup_impl(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        self.url = "https://api.deepgram.com/v1/listen"
        self.headers = {"Authorization": f"Token {self.config.api_key}"}
    
    async def transcribe_file(self, audio_path: str) -> STTResult:
        import time
        import aiohttp
        
        start = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            async with session.post(
                self.url,
                headers=self.headers,
                data=audio_data,
                params={"model": "nova-2", "language": "en", "punctuate": "true"}
            ) as resp:
                result = await resp.json()
                latency = (time.perf_counter() - start) * 1000
                
                if "results" in result:
                    channel = result["results"]["channels"][0]
                    alt = channel["alternatives"][0]
                    return STTResult(
                        text=alt["transcript"],
                        is_final=True,
                        confidence=alt.get("confidence", 0.0),
                        latency_ms=latency,
                        language=result["results"].get("language", "en"),
                    )
                
                return STTResult(text="", is_final=True, confidence=0.0, latency_ms=latency)
    
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        budget_ms: Optional[int] = None,
        reset_signal: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[STTResult]:
        import time

        # NOTE: this yields per-chunk transcripts and clears its buffer, so it
        # does NOT satisfy STTResult's cumulative-per-turn contract that
        # consumers rely on (they replace rather than append), and it ignores
        # reset_signal. Both need fixing before this provider can drive the
        # orchestrator.
        # Deepgram streaming via WebSocket would be better
        # This is simplified HTTP streaming
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            
            if len(buffer) >= 32000:  # ~1 second
                start = time.perf_counter()
                async with self.session.post(
                    self.url,
                    headers=self.headers,
                    data=buffer,
                    params={"model": "nova-2", "language": "en", "interim_results": "true"}
                ) as resp:
                    result = await resp.json()
                    latency = (time.perf_counter() - start) * 1000
                    
                    if "results" in result:
                        channel = result["results"]["channels"][0]
                        alt = channel["alternatives"][0]
                        yield STTResult(
                            text=alt["transcript"],
                            is_final=channel.get("is_final", False),
                            confidence=alt.get("confidence", 0.0),
                            latency_ms=latency,
                        )
                buffer.clear()


def create_stt(config) -> STTBase:
    """Factory function to create STT instance from config."""
    from fusion_runtime.config import Provider
    
    if config.provider == Provider.FASTER_WHISPER:
        return FasterWhisperSTT(config)
    elif config.provider == Provider.DEEPGRAM:
        return DeepgramSTT(config)
    # Add more providers...
    raise ValueError(f"Unknown STT provider: {config.provider}")