"""Well-behaved fake runtimes: no models, predictable output, adjustable delays.

They show what a correct runtime looks like, pass the conformance kit, and let
engine tests (scheduling, cancellation, barge-in) run without loading models.
Options come from ModelSpec.options:

    step_s       delay per chunk / token / utterance (default 0.01)
    words        LLM reply tokens (default "Hello from a fake model.")
    sample_rate  TTS output rate (default 24000)
    max_batch, max_concurrency, languages, voices
"""
import math
import struct
from typing import AsyncIterator, List, Optional, Sequence

from fusion_runtime.contract import (
    AdapterError,
    AudioChunk,
    Cancelled,
    Capabilities,
    InvalidRequest,
    LLMChunk,
    LLMRequest,
    LLMRuntime,
    ModelSpec,
    STTRequest,
    STTResult,
    STTRuntime,
    Transcript,
    TTSRequest,
    TTSRuntime,
)


def fake_spec(stage: str, **options) -> ModelSpec:
    return ModelSpec(stage=stage, runtime=f"fake_{stage}", model="fake", options=options)


class _FakeBase:
    def _init_fake(self) -> None:
        options = self.spec.options
        self.step_s: float = options.get("step_s", 0.01)
        self.loaded = False
        self.closed = False
        self.in_flight = 0
        self.max_in_flight = 0

    async def load(self) -> None:
        self.loaded = True

    async def close(self) -> None:
        self.closed = True

    def _enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def _exit(self) -> None:
        self.in_flight -= 1

    async def _step(self, request) -> None:
        """Wait one step, but wake immediately if the request is cancelled."""
        if await request.cancel.wait(self.step_s):
            raise Cancelled(request.cancel.reason or "cancelled")

    def _check_language(self, language: Optional[str]) -> None:
        if not self.capabilities.supports_language(language):
            raise InvalidRequest(f"language {language!r} not supported")


class FakeSTTRuntime(_FakeBase, STTRuntime):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._init_fake()

    @property
    def capabilities(self) -> Capabilities:
        o = self.spec.options
        return Capabilities(
            streaming_output=False,
            sample_rate=16000,
            languages=tuple(o["languages"]) if "languages" in o else None,
            max_batch=o.get("max_batch", 4),
            max_concurrency=o.get("max_concurrency", 2),
        )

    async def transcribe(self, requests: Sequence[STTRequest]) -> List[STTResult]:
        self._enter()
        try:
            return [await self._one(r) for r in requests]
        finally:
            self._exit()

    async def _one(self, request: STTRequest) -> STTResult:
        try:
            if not request.audio or len(request.audio) % 2:
                raise InvalidRequest("audio must be non-empty 16-bit PCM")
            self._check_language(request.language)
            request.cancel.raise_if_cancelled()
            await self._step(request)
            samples = len(request.audio) // 2
            return Transcript(
                text=f"{samples} samples",
                language=request.language or "en",
                confidence=1.0,
                duration_s=samples / request.sample_rate,
            )
        except AdapterError as e:
            return e


class FakeLLMRuntime(_FakeBase, LLMRuntime):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._init_fake()
        self.words: List[str] = list(spec.options.get("words", ["Hello", " from", " a", " fake", " model."]))
        self.decoded = 0

    @property
    def capabilities(self) -> Capabilities:
        o = self.spec.options
        return Capabilities(
            max_concurrency=o.get("max_concurrency", 2),
            languages=tuple(o["languages"]) if "languages" in o else None,
            tools=True,
        )

    async def generate(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        if not request.messages:
            raise InvalidRequest("messages must not be empty")
        self._check_language(request.language)
        request.cancel.raise_if_cancelled()
        self._enter()
        try:
            words = self.words[: max(1, request.max_tokens)]
            for i, word in enumerate(words):
                await self._step(request)  # a "decode step", only when the caller pulls
                self.decoded += 1
                last = i == len(words) - 1
                yield LLMChunk(
                    text=word,
                    finish_reason="stop" if last else None,
                    usage={"completion_tokens": i + 1} if last else {},
                )
        finally:
            self._exit()


class FakeTTSRuntime(_FakeBase, TTSRuntime):
    CHUNK_MS = 40

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._init_fake()

    @property
    def capabilities(self) -> Capabilities:
        o = self.spec.options
        return Capabilities(
            sample_rate=o.get("sample_rate", 24000),
            voices=tuple(o.get("voices", ("fake_voice",))),
            languages=tuple(o["languages"]) if "languages" in o else None,
            max_concurrency=o.get("max_concurrency", 2),
        )

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if not request.text.strip():
            raise InvalidRequest("text must not be empty")
        if request.voice is not None and request.voice not in self.capabilities.voices:
            raise InvalidRequest(f"unknown voice {request.voice!r}")
        self._check_language(request.language)
        request.cancel.raise_if_cancelled()
        rate = self.capabilities.sample_rate
        samples_per_chunk = rate * self.CHUNK_MS // 1000
        chunks = max(1, len(request.text) // 8)  # longer text, more audio
        self._enter()
        try:
            for index in range(chunks):
                await self._step(request)
                tone = (int(3000 * math.sin(2 * math.pi * 220 * (index * samples_per_chunk + n) / rate))
                        for n in range(samples_per_chunk))
                yield AudioChunk(pcm=struct.pack(f"<{samples_per_chunk}h", *tone), sample_rate=rate)
        finally:
            self._exit()
