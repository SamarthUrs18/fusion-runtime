"""Speech to text behind any OpenAI-compatible endpoint: POST /audio/transcriptions.

One adapter for the protocol, not one per vendor: vLLM serving a Whisper model
(which batches callers on its side, on your GPU), OpenAI with a key, and any other
server that speaks the same API. The model reference is the base URL, for example
http://localhost:8002/v1 or https://api.openai.com/v1.

Each window the engine transcribes goes up as a 16 kHz mono WAV file; the reply's
`text` is the transcript. Keys never live in config: `api_key_env` names the
environment variable that holds one, and without it no Authorization header is sent.

Options:
    model_name       the model's name on the server (required), e.g. openai/whisper-large-v3, whisper-1
    api_key_env      environment variable holding the API key
    timeout_s        connect/read timeout (default 30)
    verify           check the endpoint and key while loading (default true)
    max_concurrency  windows in flight at once (default 16)
    extra_body       extra form fields sent with every request (temperature, server-specific settings)
    language         what callers speak, when a request doesn't say (from STTConfig)
"""
import asyncio
import io
import wave
from typing import List, Optional, Sequence

import httpx

from fusion_runtime.contract import (
    AdapterError,
    AuthFailed,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    RuntimeFailure,
    STTRequest,
    STTResult,
    STTRuntime,
    Transcript,
)
from fusion_runtime.runtimes.openai_http.llm import (
    _client,
    _connection_error,
    _status_error,
    api_key,
    probe_endpoint,
)

OPTIONS = ("model_name", "api_key_env", "timeout_s", "verify", "max_concurrency", "extra_body", "language")
SAMPLE_RATE = 16000


class OpenAIHTTPSTT(STTRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.client: Optional[httpx.AsyncClient] = None
        self.transport: Optional[httpx.AsyncBaseTransport] = None  # tests plug in a fake server here
        self._health = Health("down", "not loaded")

    @property
    def model_name(self) -> Optional[str]:
        return self.spec.options.get("model_name")

    @property
    def api_key_env(self) -> Optional[str]:
        return self.spec.options.get("api_key_env")

    @property
    def capabilities(self) -> Capabilities:
        # One window per request: the server batches them itself if it can (vLLM does)
        return Capabilities(streaming_output=False, sample_rate=SAMPLE_RATE,
                            max_concurrency=self.spec.options.get("max_concurrency", 16))

    def health(self) -> Health:
        return self._health

    async def load(self) -> None:
        await _connect(self, "stt")

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        self._health = Health("down", "closed")

    async def transcribe(self, requests: Sequence[STTRequest]) -> List[STTResult]:
        return list(await asyncio.gather(*(self._one(request) for request in requests)))

    async def _one(self, request: STTRequest) -> STTResult:
        try:
            if request.cancel.cancelled:
                raise Cancelled(request.cancel.reason or "cancelled")
            if not request.audio or len(request.audio) % 2:
                raise InvalidRequest("audio must be non-empty 16-bit PCM")
            if self.client is None:
                raise RuntimeFailure("not loaded")
            language = request.language or self.spec.options.get("language")
            form = {k: str(v) for k, v in (self.spec.options.get("extra_body") or {}).items()}
            form.update(model=self.model_name, response_format="json")
            if language:
                form["language"] = language.split("-")[0].lower()  # the API takes ISO-639-1 ("en", not "en-US")
            if request.prompt:
                form["prompt"] = request.prompt
            files = {"file": ("audio.wav", _wav(request.audio, request.sample_rate), "audio/wav")}
            response = await _until_cancelled(
                request, self.client.post("/audio/transcriptions", data=form, files=files), self.spec.model)
            if not response.is_success:
                raise _status_error(response, self.spec.model)
            try:
                body = response.json()
                text = body["text"]
            except (ValueError, KeyError, TypeError):
                raise RuntimeFailure(f"{self.spec.model} answered without a transcript: {response.text[:200]}")
            return Transcript(text=text, language=body.get("language") or language,
                              duration_s=len(request.audio) / 2 / request.sample_rate)
        except AdapterError as e:
            return e


async def _connect(runtime, stage: str) -> None:
    """Check the model name, the key and (unless verify=False) the endpoint, then open the client."""
    from fusion_runtime.telemetry import telemetry

    url = runtime.spec.model
    if not runtime.model_name:
        raise InvalidRequest(f"set the model name to use on {url} (model_name)")
    if runtime.api_key_env and not api_key(runtime.api_key_env):
        raise AuthFailed(f"the API key for {url} is missing: set the environment variable {runtime.api_key_env}")
    timeout = runtime.spec.options.get("timeout_s", 30.0)
    if runtime.spec.options.get("verify", True):
        report = await probe_endpoint(url, runtime.api_key_env, min(timeout, 10.0), runtime.transport)
        if report.error is not None:
            raise report.error
        if report.lists(runtime.model_name) is False:
            telemetry.emit(f"{stage}.endpoint_model_unlisted", level="warning", stage=stage, url=url,
                           model=runtime.model_name, listed=report.models[:10],
                           hint="check model_name if requests fail with model_not_found")
    runtime.client = _client(url, runtime.api_key_env, timeout, runtime.transport)
    runtime._health = Health("ok")


async def _until_cancelled(request, work, url: str):
    """Await `work`, but stop as soon as the request's CancelToken fires (from any thread)."""
    task = asyncio.ensure_future(work)
    loop = asyncio.get_running_loop()

    def stop() -> None:
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # loop closed
            pass

    request.cancel.add_callback(stop)
    try:
        return await task
    except asyncio.CancelledError:
        if request.cancel.cancelled:
            raise Cancelled(request.cancel.reason or "cancelled") from None
        raise
    except httpx.HTTPError as e:
        raise _connection_error(url, e) from e


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
