"""faster-whisper STT engine (CTranslate2 backend)."""
from typing import AsyncIterator, Optional
import asyncio

from fusion_runtime.stt.base import STTBase, STTResult


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
    
    def _transcribe_sync(self, audio, **options):
        """Transcribe and fully decode. Run in a worker thread.

        faster-whisper's transcribe() returns a lazy generator: the decoding
        happens while iterating the segments. Iterating them back on the event
        loop froze the whole server for ~200 ms per window, so the text is
        joined here, inside the thread.
        """
        segments, info = self.model.transcribe(audio, **options)
        return " ".join(segment.text for segment in segments), info

    async def _warmup_impl(self):
        await asyncio.get_running_loop().run_in_executor(None, self._load_and_warm_sync)

    def _load_and_warm_sync(self):
        from faster_whisper import WhisperModel
        
        # Resolve device/compute_type ("auto" adapts to the machine)
        device = self.config.device
        compute_type = self.config.compute_type
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        from fusion_runtime.config import resolve_model_path
        local = resolve_model_path(f"stt/{self.config.model}")
        model = str(local) if (local / "model.bin").exists() else self.config.model

        self.model = WhisperModel(
            model,
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
        self._transcribe_sync(buf, language=self.config.language)
    
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
                
                # Run in thread pool to avoid blocking (decoding included)
                loop = asyncio.get_running_loop()
                text, info = await loop.run_in_executor(
                    None,
                    lambda: self._transcribe_sync(
                        audio_np,
                        language=self.config.language,
                        beam_size=self.config.beam_size,
                        vad_filter=self.config.vad_filter,
                        condition_on_previous_text=False,  # Critical for streaming
                    )
                )
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
            
            loop = asyncio.get_running_loop()
            text, info = await loop.run_in_executor(
                None,
                lambda: self._transcribe_sync(
                    audio_np,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                )
            )
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
        loop = asyncio.get_running_loop()
        text, info = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(
                audio_np,
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
            )
        )
        latency = (time.perf_counter() - start) * 1000
        
        return STTResult(
            text=text,
            is_final=True,
            confidence=info.language_probability,
            latency_ms=latency,
            language=info.language,
        )
