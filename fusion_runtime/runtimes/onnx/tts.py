"""Text-to-speech ONNX models, with the model family supplying input preparation.

The runtime handles loading, threading, cancellation, voices and errors; a
family spec (see FAMILIES) knows one model family's text processing and
tensor layout. A new ONNX TTS family is a new spec, not a new runtime.

Options (from config): voice (default voice), speed, voices_path, warmup
(default true), plus whatever the family reads.
"""
import asyncio
import importlib
from pathlib import Path
from typing import AsyncIterator

from fusion_runtime.contract import (
    AudioChunk,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    RuntimeFailure,
    TTSRequest,
    TTSRuntime,
    UnsupportedModel,
)

FAMILIES = {
    "kokoro": "fusion_runtime.runtimes.onnx.kokoro:KokoroFamily",
}


class OnnxTTS(TTSRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        target = FAMILIES.get(spec.family or "")
        if target is None:
            known = ", ".join(sorted(FAMILIES))
            raise UnsupportedModel(f"no ONNX TTS family {spec.family!r}; known families: {known}")
        module, _, name = target.partition(":")
        self.family = getattr(importlib.import_module(module), name)(Path(spec.model), spec.options)
        self._loaded = False

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming_output=True,
            sample_rate=self.family.sample_rate,
            voices=self.family.voices,
            languages=self.family.languages if self._loaded else None,
            max_concurrency=self.spec.options.get("max_concurrency", 1),
        )

    def health(self) -> Health:
        return Health("ok") if self._loaded else Health("down", "model not loaded")

    @property
    def default_voice(self) -> str:
        voices = self.family.voices
        wanted = self.spec.options.get("voice")
        return wanted if wanted in voices else (voices[0] if voices else "")

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.family.load)
        self._loaded = True
        voice = self.spec.options.get("voice")
        if voice and voice not in self.family.voices:
            raise InvalidRequest(f"voice {voice!r} isn't in this model; available: {', '.join(self.family.voices)}")
        if self.spec.options.get("warmup", True):
            async for _ in self.synthesize(TTSRequest(text="Warmup.")):
                pass

    async def close(self) -> None:
        self._loaded = False

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if not request.text.strip():
            raise InvalidRequest("text must not be empty")
        voice = request.voice or self.default_voice
        if voice not in self.family.voices:
            raise InvalidRequest(f"unknown voice {voice!r}; available: {', '.join(self.family.voices)}")
        if request.language and not self.capabilities.supports_language(request.language):
            raise InvalidRequest(f"language {request.language!r} isn't supported by these voices")
        request.cancel.raise_if_cancelled()
        if not self._loaded:
            raise RuntimeFailure("model not loaded")
        try:
            pcm = await asyncio.get_running_loop().run_in_executor(
                None, self.family.synthesize, request.text, voice, request.speed, request.language)
        except Exception as e:
            raise RuntimeFailure(f"{self.spec.family} synthesis failed: {e}") from e
        if request.cancel.cancelled:
            raise Cancelled(request.cancel.reason or "cancelled")
        yield AudioChunk(pcm=pcm, sample_rate=self.family.sample_rate)
