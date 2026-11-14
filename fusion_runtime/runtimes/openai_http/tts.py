"""Text to speech behind any OpenAI-compatible endpoint: POST /audio/speech.

One adapter for the protocol, not one per vendor: OpenAI with a key, Kokoro-FastAPI,
vLLM-served speech models and other servers that speak the same API. The model
reference is the base URL, for example https://api.openai.com/v1.

Audio is asked for as raw PCM (`response_format: pcm`, 16-bit mono, which OpenAI
sends at 24 kHz) and streamed on as it arrives, so the caller hears the first
words before the server has finished the sentence. A server that only sends WAV
works with response_format="wav": its header is read for the sample rate.

Options:
    model_name       the model's name on the server (required), e.g. tts-1, gpt-4o-mini-tts, kokoro
    api_key_env      environment variable holding the API key
    timeout_s        connect/read timeout (default 30)
    verify           check the endpoint and key while loading (default true)
    max_concurrency  sentences in flight at once (default 16)
    extra_body       extra fields merged into every request (instructions, server-specific settings)
    response_format  pcm (default) or wav
    sample_rate      Hz of the server's raw PCM (default 24000, OpenAI's); read from the header for wav
    voice, speed     from TTSConfig: the voice name on the server, and how fast it speaks
"""
import asyncio
import struct
from typing import AsyncIterator, List, Optional

import httpx

from fusion_runtime.contract import (
    AudioChunk,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    RuntimeFailure,
    TTSRequest,
    TTSRuntime,
)
from fusion_runtime.runtimes.openai_http.llm import _connection_error, _status_error
from fusion_runtime.runtimes.openai_http.stt import _connect

OPTIONS = ("model_name", "api_key_env", "timeout_s", "verify", "max_concurrency", "extra_body",
           "response_format", "sample_rate", "voice", "speed")
FORMATS = ("pcm", "wav")
OPENAI_PCM_RATE = 24000


class OpenAIHTTPTTS(TTSRuntime):
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
    def response_format(self) -> str:
        return self.spec.options.get("response_format", "pcm")

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(streaming_output=True,
                            sample_rate=int(self.spec.options.get("sample_rate") or OPENAI_PCM_RATE),
                            max_concurrency=self.spec.options.get("max_concurrency", 16))

    def health(self) -> Health:
        return self._health

    async def load(self) -> None:
        if self.response_format not in FORMATS:
            raise InvalidRequest(f"response_format must be one of {', '.join(FORMATS)}, got {self.response_format!r}: "
                                 "audio has to arrive uncompressed to be streamed to the caller")
        await _connect(self, "tts")

    async def close(self) -> None:
        client, self.client = self.client, None
        self._health = Health("down", "closed")
        if client is not None:
            await client.aclose()

    async def synthesize(self, request: TTSRequest) -> AsyncIterator[AudioChunk]:
        if not request.text.strip():
            raise InvalidRequest("text must not be empty")
        request.cancel.raise_if_cancelled()
        if self.client is None:
            raise RuntimeFailure("not loaded")
        body = {**(self.spec.options.get("extra_body") or {}), "model": self.model_name, "input": request.text,
                "voice": request.voice or self.spec.options.get("voice"), "response_format": self.response_format,
                "speed": request.speed}
        reader = _PCMReader(self.response_format, self.capabilities.sample_rate)
        loop = asyncio.get_running_loop()
        try:
            async with self.client.stream("POST", "/audio/speech", json=body) as response:
                def close_on_cancel() -> None:
                    # Wakes a read waiting on a slow server; the loop below then raises Cancelled
                    try:
                        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(response.aclose()))
                    except RuntimeError:
                        pass

                request.cancel.add_callback(close_on_cancel)
                if not response.is_success:
                    await response.aread()
                    raise _status_error(response, self.spec.model)
                async for data in response.aiter_bytes():
                    if request.cancel.cancelled:
                        break
                    for pcm in reader.feed(data):
                        yield AudioChunk(pcm=pcm, sample_rate=reader.sample_rate)
        except httpx.HTTPError as e:
            if request.cancel.cancelled:
                raise Cancelled(request.cancel.reason or "cancelled") from None
            raise _connection_error(self.spec.model, e) from e
        if request.cancel.cancelled:
            raise Cancelled(request.cancel.reason or "cancelled")
        if not reader.started:
            raise RuntimeFailure(f"{self.spec.model} sent no audio for {len(request.text)} characters of text")


class _PCMReader:
    """Bytes from the server in, whole 16-bit samples out; a WAV header is read and dropped."""

    def __init__(self, response_format: str, sample_rate: int):
        self.wav = response_format == "wav"
        self.sample_rate = sample_rate
        self._pending = b""
        self._in_header = self.wav
        self.started = False

    def feed(self, data: bytes) -> List[bytes]:
        self._pending += data
        if self._in_header and not self._read_header():
            return []
        usable = len(self._pending) - len(self._pending) % 2  # never split a sample
        if not usable:
            return []
        pcm, self._pending = self._pending[:usable], self._pending[usable:]
        self.started = True
        return [pcm]

    def _read_header(self) -> bool:
        """Walk the RIFF chunks up to "data". False until enough of the header has arrived."""
        buffer = self._pending
        if len(buffer) < 12:
            return False
        if buffer[:4] != b"RIFF" or buffer[8:12] != b"WAVE":
            raise RuntimeFailure("the server's audio isn't a WAV file; use response_format=\"pcm\"")
        offset = 12
        while True:
            if len(buffer) < offset + 8:
                return False
            name, size = buffer[offset:offset + 4], struct.unpack("<I", buffer[offset + 4:offset + 8])[0]
            if name == b"data":
                self._pending = buffer[offset + 8:]
                self._in_header = False
                return True
            if len(buffer) < offset + 8 + size:
                return False
            if name == b"fmt ":
                channels, rate, _, _, bits = struct.unpack("<HIIHH", buffer[offset + 10:offset + 24])
                if channels != 1 or bits != 16:
                    raise RuntimeFailure(f"the server sent {channels}-channel {bits}-bit audio; fusion needs mono 16-bit")
                self.sample_rate = rate
            offset += 8 + size + (size % 2)
