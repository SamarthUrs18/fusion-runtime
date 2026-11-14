"""Any Whisper-family model converted to CTranslate2, through faster-whisper.

Covers every Whisper size and language (tiny to large-v3, distil-whisper,
fine-tunes) with the same code. The engine decides when to transcribe
(rolling windows, end of speech); this runtime only turns utterances into text.

Options (from config): device (auto | cpu | cuda), compute_type (auto | int8 |
float16 | ...), beam_size, vad_filter, language (default when a request has
none), cpu_threads, warmup (default true), no_speech_threshold, max_batch.

Batches: callers' windows that arrive while the model is busy go through in one
CTranslate2 call (the encoder and decoder run once for all of them), the way
faster-whisper's own BatchedInferencePipeline does it. `max_batch` defaults to
8 on a GPU and 1 on the CPU, where a batch gains little and one caller must
not get slower. A batch of one takes the usual path.
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

# Settings this runtime reads (the resolver refuses others, so a typo doesn't pass silently)
OPTIONS = ("device", "compute_type", "language", "beam_size", "vad_filter", "no_speech_threshold", "cpu_threads",
           "warmup", "max_batch")

DEFAULT_GPU_BATCH = 8

WHISPER_SAMPLE_RATE = 16000

# Whisper invents text when it hears almost nothing — usually phrases from the
# videos it was trained on ("Thank you for watching", "We'll see you guys").
# It reports how sure it is that a segment is *not* speech, and those inventions
# score high: measured on this pipeline, real speech is 0.01-0.2, while faint
# leaked audio that produced a hallucination was 0.68. Segments above this are
# dropped rather than answered.
DEFAULT_NO_SPEECH_THRESHOLD = 0.6


class CTranslate2STT(STTRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.model = None
        self._languages: Optional[tuple] = None
        self._max_batch = 1

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming_input=False,
            streaming_output=False,
            sample_rate=WHISPER_SAMPLE_RATE,
            languages=self._languages,
            max_batch=self._max_batch,
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
        self._max_batch = self._batch_size()
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

    def _batch_size(self) -> int:
        wanted = self.spec.options.get("max_batch")
        if wanted is not None:
            if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
                raise InvalidRequest(f"max_batch must be a whole number of at least 1, got {wanted!r}")
            return wanted
        on_gpu = getattr(getattr(self.model, "model", None), "device", "cpu") == "cuda"
        return DEFAULT_GPU_BATCH if on_gpu else 1

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
        joined here, inside the thread. Segments the model itself says probably
        aren't speech are dropped (see DEFAULT_NO_SPEECH_THRESHOLD).
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
        threshold = options.get("no_speech_threshold", DEFAULT_NO_SPEECH_THRESHOLD)
        kept, dropped = [], []
        for segment in segments:
            score = getattr(segment, "no_speech_prob", None)
            if threshold is not None and score is not None and score > threshold:
                dropped.append((segment.text, score))
            else:
                kept.append(segment.text)
        return " ".join(kept), info, dropped

    def _transcribe_batch_sync(self, items: Sequence[tuple]) -> List[tuple]:
        """Several windows in one model call: (pcm, language, prompt) each. Runs in a worker thread.

        Mirrors _transcribe_sync for windows of up to 30 s (the engine's limit): the same
        features, prompt, greedy or beam decoding and no-speech check, with the encoder and
        decoder run once for the whole batch. Returns (text, language, language_probability,
        dropped) per window.
        """
        import numpy as np
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        model, options = self.model, self.spec.options
        multilingual = model.model.is_multilingual
        features = []
        for pcm, _, _ in items:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            audio = audio[-model.feature_extractor.n_samples:]  # the latest 30 s, as the engine keeps
            features.append(pad_or_trim(model.feature_extractor(audio)[..., :-1]))
        encoder_output = model.encode(np.stack(features))

        languages = [language if multilingual else "en" for _, language, _ in items]
        probabilities = [1.0] * len(items)
        if multilingual and any(language is None for language in languages):
            detected = model.model.detect_language(encoder_output)
            for i, language in enumerate(languages):
                if language is None:
                    token, probabilities[i] = detected[i][0]
                    languages[i] = token[2:-2]  # "<|en|>" -> "en"

        tokenizers, prompts = [], []
        for (_, _, prompt), language in zip(items, languages):
            tokenizer = Tokenizer(model.hf_tokenizer, multilingual, task="transcribe", language=language)
            previous = tokenizer.encode(" " + prompt.strip()) if prompt else []
            tokenizers.append(tokenizer)
            prompts.append(model.get_prompt(tokenizer, previous, without_timestamps=True))

        results = model.model.generate(
            encoder_output, prompts,
            beam_size=options.get("beam_size", 1),
            max_length=model.max_length,
            suppress_blank=True,
            suppress_tokens=list(get_suppressed_tokens(tokenizers[0], [-1])),
            return_scores=True,
            return_no_speech_prob=True,
        )
        threshold = options.get("no_speech_threshold", DEFAULT_NO_SPEECH_THRESHOLD)
        out = []
        for result, tokenizer, language, probability in zip(results, tokenizers, languages, probabilities):
            text = tokenizer.decode(result.sequences_ids[0])
            score = result.no_speech_prob
            if threshold is not None and score > threshold and text.strip():
                out.append(("", language, probability, [(text, score)]))
            else:
                out.append((text, language, probability, []))
        return out

    async def transcribe(self, requests: Sequence[STTRequest]) -> List[STTResult]:
        if len(requests) <= 1 or self._max_batch <= 1:
            return [await self._one(request) for request in requests]
        results: List[Optional[STTResult]] = [None] * len(requests)
        batch = []
        for i, request in enumerate(requests):
            try:
                batch.append((i, request, self._check(request)))
            except AdapterError as e:
                results[i] = e
        if len(batch) == 1:
            i, request, _ = batch[0]
            results[i] = await self._one(request)
        elif batch:
            items = [(request.audio, _whisper_language(language), request.prompt) for _, request, language in batch]
            try:
                outs = await asyncio.get_running_loop().run_in_executor(None, self._transcribe_batch_sync, items)
            except Exception as e:
                failure = RuntimeFailure(f"Whisper transcription failed ({len(batch)} in one batch): {e}")
                for i, _, _ in batch:
                    results[i] = failure
            else:
                for (i, request, _), (text, language, probability, dropped) in zip(batch, outs):
                    try:
                        results[i] = self._finish(request, text, language, probability, dropped)
                    except AdapterError as e:
                        results[i] = e
        return results

    def _check(self, request: STTRequest) -> Optional[str]:
        """The language to use, or raises what's wrong with the request."""
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
        return language

    def _finish(self, request: STTRequest, text: str, language, probability, dropped) -> Transcript:
        if dropped:
            from fusion_runtime.telemetry import telemetry

            telemetry.emit(
                "stt.invented_speech_dropped", level="debug", stage="stt", session_id=request.session_id,
                request_id=request.id, segments=len(dropped),
                no_speech_prob=round(max(score for _, score in dropped), 3),
                hint="the model was mostly sure this wasn't speech, so its guess wasn't answered",
            )
        if request.cancel.cancelled:  # the result arrived after nobody wanted it
            raise Cancelled(request.cancel.reason or "cancelled")
        return Transcript(text=text, language=language, confidence=probability,
                          duration_s=len(request.audio) / 2 / request.sample_rate)

    async def _one(self, request: STTRequest) -> STTResult:
        try:
            language = self._check(request)
            try:
                text, info, dropped = await asyncio.get_running_loop().run_in_executor(
                    None, self._transcribe_sync, request.audio, _whisper_language(language), request.prompt)
            except Exception as e:
                raise RuntimeFailure(f"Whisper transcription failed: {e}") from e
            return self._finish(request, text, getattr(info, "language", None),
                                getattr(info, "language_probability", None), dropped)
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
