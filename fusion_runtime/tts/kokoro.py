"""Kokoro TTS engine (ONNX)."""
from typing import AsyncIterator, Optional, List
import asyncio
import numpy as np

from fusion_runtime.tts.base import TTSBase, TTSResult


class KokoroTTS(TTSBase):
    """Kokoro ONNX implementation.

    Uses kokoro-onnx's bundled tokenizer (espeak data included — no system
    espeak needed) and calls the ONNX session directly for full control
    over the style-vector shape.

    Files needed, relative to the model dir (`frun models pull --kokoro`):
      - tts/onnx/model.onnx
      - tts/voices-v1.0.bin   (assembled voice pack)
    """
    
    SAMPLE_RATE = 24000
    
    def __init__(self, config):
        super().__init__(config)
        self._kokoro = None
        self._voices_path = getattr(config, "voices_path", None)
    
    async def _warmup_impl(self):
        from pathlib import Path
        from kokoro_onnx import Kokoro
        from fusion_runtime.config import resolve_model_path
        
        model_path = resolve_model_path(self.config.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Kokoro model not found at {model_path}. "
                "Run: frun models pull --kokoro"
            )
        
        # Resolve voices pack: explicit path, else next to model or one level up
        candidates = []
        if self._voices_path:
            candidates.append(resolve_model_path(self._voices_path))
        candidates += [
            model_path.parent / "voices-v1.0.bin",
            model_path.parent.parent / "voices-v1.0.bin",
        ]
        voices = next((p for p in candidates if p.exists()), None)
        if voices is None:
            raise FileNotFoundError(
                f"Kokoro voices pack (voices-v1.0.bin) not found in any of: "
                f"{[str(c) for c in candidates]}. "
                "Run: frun models pull --kokoro"
            )
        self._voices_path = str(voices)
        
        # Official wrapper: gives us tokenizer + session + voices,
        # but we run inference ourselves for correct style rank.
        self._kokoro = Kokoro(str(model_path), self._voices_path)
        
        # Warmup inference
        await self.synthesize("Warmup.")
    
    def _infer_sync(self, text: str) -> tuple[bytes, int]:
        """Synchronous inference — call via run_in_executor."""
        k = self._kokoro
        
        # 1. Phonemize + tokenize (bundled espeak data, no system dep)
        phonemes = k.tokenizer.phonemize(text, "en-us")
        tokens = k.tokenizer.tokenize(phonemes)
        if not tokens:
            # Whitespace/punctuation-only chunk → tiny silence
            return (np.zeros(240, dtype=np.int16).tobytes(), self.SAMPLE_RATE)
        
        # 2. Style vector: row (len-1) of the voice pack, reshaped to [1, 256]
        voice = k.voices[self.config.voice]
        style = voice[min(len(tokens), len(voice)) - 1].reshape(1, 256).astype(np.float32)
        
        # 3. Run session
        outputs = k.sess.run(None, {
            "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
            "style": style,
            "speed": np.array([self.config.speed], dtype=np.float32),
        })
        audio = outputs[0].ravel()
        
        # 4. Float32 → int16 PCM
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        return pcm, self.SAMPLE_RATE
    
    async def _synthesize_chunk(self, text: str) -> tuple[bytes, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, text)
    
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        budget_ms: Optional[int] = None
    ) -> AsyncIterator[TTSResult]:
        import time
        
        buffer = ""
        chunk_chars = 100  # Synthesize every ~100 chars
        
        async for text_chunk in text_stream:
            buffer += text_chunk
            
            if len(buffer) >= chunk_chars or text_chunk.endswith((".", "!", "?", "\n")):
                start = time.perf_counter()
                audio, _ = await self._synthesize_chunk(buffer)
                latency = (time.perf_counter() - start) * 1000
                
                yield TTSResult(
                    audio=audio,
                    is_final=False,
                    sample_rate=self.config.sample_rate,
                    latency_ms=latency,
                    format="pcm",
                )
                buffer = ""
        
        # Flush remaining
        if buffer:
            start = time.perf_counter()
            audio, _ = await self._synthesize_chunk(buffer)
            yield TTSResult(
                audio=audio,
                is_final=True,
                sample_rate=self.config.sample_rate,
                latency_ms=(time.perf_counter() - start) * 1000,
                format="pcm",
            )
    
    async def synthesize(self, text: str) -> TTSResult:
        import time
        start = time.perf_counter()
        audio, sr = await self._synthesize_chunk(text)
        return TTSResult(
            audio=audio,
            is_final=True,
            sample_rate=sr,
            latency_ms=(time.perf_counter() - start) * 1000,
            format="pcm",
        )
