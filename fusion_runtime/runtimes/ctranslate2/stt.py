"""Any Whisper-family model converted to CTranslate2, through faster-whisper.

Covers every Whisper size and language (tiny to large-v3, distil-whisper,
fine-tunes) with the same code. The engine decides when to transcribe
(rolling windows, end of speech); this runtime only turns utterances into text.

Options (from config): device (auto | cpu | cuda), compute_type (auto | int8 |
float16 | ...), beam_size, vad_filter, language (default when a request has
none), cpu_threads, warmup (default true).
"""
import asyncio
from typing import List, Optional, Sequence

from fusion_runtime.contract import (
    AdapterError,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    ModelNotFound,
    RuntimeFailure,
    STTRequest,
    STTResult,
    STTRuntime,
    Transcript,
)

WHISPER_SAMPLE_RATE = 16000


class CTranslate2STT(STTRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.model = None
        self._languages: Optional[tuple] = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming_input=False,
            streaming_output=False,
            sample_rate=WHISPER_SAMPLE_RATE,
            languages=self._languages,
            max_batch=1,
            max_concurrency=1,  # one CTranslate2 worker; more workers are a future option
        )

    def health(self) -> Health:
        return Health("ok") if self.model is not None else Health("down", "model not loaded")

    async def load(self) -> None:
        from pathlib import Path

        path = Path(self.spec.model)
        if not (path / "model.bin").is_file():
            raise ModelNotFound(f"Whisper model not found at {path}. Run: frun models pull")
        self.model = await asyncio.get_running_loop().run_in_executor(None, self._build, str(path))
        self._languages = self._detect_languages()
        language = self._default_language()
        if not self.capabilities.supports_language(language):
            raise InvalidRequest(
                f"{path.name} is an English-only Whisper model, but the configured language is {language!r}. "
                "Use a multilingual Whisper model (without .en in its name), or set language=\"en\""
            )
        if self.spec.options.get("warmup", True):
            await asyncio.get_running_loop().run_in_executor(
                None, self._transcribe_sync, b"\x00\x00" * WHISPER_SAMPLE_RATE, self._default_language())

    async def close(self) -> None:
        self.model = None

    def _build(self, path: str):
        from faster_whisper import WhisperModel

        options = self.spec.options
        device = options.get("device", "auto")
        compute_type = options.get("compute_type", "auto")
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        kwargs = {"device": device, "compute_type": compute_type}
        if options.get("cpu_threads"):
            kwargs["cpu_threads"] = options["cpu_threads"]
        return WhisperModel(path, **kwargs)

    def _detect_languages(self) -> Optional[tuple]:
        if getattr(self.model, "model", None) is not None and not getattr(self.model.model, "is_multilingual", True):
            return ("en",)
        return None

    def _default_language(self) -> Optional[str]:
        return self.spec.options.get("language")

    def _transcribe_sync(self, pcm: bytes, language: Optional[str], prompt: Optional[str] = None):
        """Transcribe and fully decode. Runs in a worker thread.

        faster-whisper's transcribe() returns a lazy generator: the decoding
        happens while iterating the segments. Iterating them back on the event
        loop froze the whole server for ~200 ms per window, so the text is
        joined here, inside the thread.
        """
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        options = self.spec.options
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=options.get("beam_size", 1),
            vad_filter=options.get("vad_filter", True),
            condition_on_previous_text=False,  # each call is one self-contained window
            initial_prompt=prompt,
        )
        return " ".join(segment.text for segment in segments), info

    async def transcribe(self, requests: Sequence[STTRequest]) -> List[STTResult]:
        results: List[STTResult] = []
        for request in requests:
            results.append(await self._one(request))
        return results

    async def _one(self, request: STTRequest) -> STTResult:
        try:
            if request.cancel.cancelled:
                raise Cancelled(request.cancel.reason or "cancelled")
            if not request.audio or len(request.audio) % 2:
                raise InvalidRequest("audio must be non-empty 16-bit PCM")
            if request.sample_rate != WHISPER_SAMPLE_RATE:
                raise InvalidRequest(f"Whisper needs {WHISPER_SAMPLE_RATE} Hz audio, got {request.sample_rate} Hz")
            language = request.language or self._default_language()
            if not self.capabilities.supports_language(language):
                raise InvalidRequest(f"this model doesn't support language {language!r} (English only)")
            if self.model is None:
                raise RuntimeFailure("model not loaded")
            try:
                text, info = await asyncio.get_running_loop().run_in_executor(
                    None, self._transcribe_sync, request.audio, _whisper_language(language), request.prompt)
            except Exception as e:
                raise RuntimeFailure(f"Whisper transcription failed: {e}") from e
            if request.cancel.cancelled:  # the result arrived after nobody wanted it
                raise Cancelled(request.cancel.reason or "cancelled")
            return Transcript(
                text=text,
                language=getattr(info, "language", None),
                confidence=getattr(info, "language_probability", None),
                duration_s=len(request.audio) / 2 / request.sample_rate,
            )
        except AdapterError as e:
            return e


def _whisper_language(language: Optional[str]) -> Optional[str]:
    """Whisper takes base codes ("en", "hi"), not regional ones ("en-US")."""
    return language.split("-")[0].lower() if language else None


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
