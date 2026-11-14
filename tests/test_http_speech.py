"""Speech behind OpenAI-compatible endpoints: /audio/transcriptions and /audio/speech."""
import asyncio
import io
import json
import struct
import wave

import httpx
import pytest
from fusion_runtime.agent import STT, TTS, Agent
from fusion_runtime.contract import (
    AuthFailed,
    Cancelled,
    InvalidRequest,
    ModelSpec,
    RateLimited,
    RuntimeFailure,
    STTRequest,
    Transcript,
    TTSRequest,
)
from fusion_runtime.contract.common import CancelToken
from fusion_runtime.resolver import resolve_stage_config
from fusion_runtime.runtimes.openai_http.stt import OpenAIHTTPSTT
from fusion_runtime.runtimes.openai_http.tts import OpenAIHTTPTTS
from fusion_runtime.testing.conformance import assert_conforms, check_runtime

URL = "http://speech.test/v1"
PCM = struct.pack("<1600h", *([1200, -1200] * 800))  # 0.1 s at 16 kHz


class FakeSpeechServer:
    """/models, /audio/transcriptions (multipart) and /audio/speech (streamed PCM or WAV)."""

    def __init__(self, text="Where is my order?", key=None, status=200, wav=False, chunks=4, step_s=0.0,
                 models=("whisper", "voice")):
        self.text, self.key, self.status, self.wav, self.chunks, self.step_s = text, key, status, wav, chunks, step_s
        self.models = models
        self.uploads = []  # (form fields, the uploaded WAV's frames and rate)
        self.speech = []  # JSON bodies

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.key and request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": {"message": "invalid api key"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if self.status != 200:
            headers = {"retry-after": "3"} if self.status == 429 else {}
            return httpx.Response(self.status, json={"error": {"message": "nope"}}, headers=headers)
        if request.url.path.endswith("/audio/transcriptions"):
            return self._transcribe(request)
        return self._speak(request)

    def _transcribe(self, request):
        body = request.read()
        boundary = request.headers["content-type"].split("boundary=")[1].encode()
        fields, audio = {}, None
        for part in body.split(b"--" + boundary)[1:-1]:
            head, _, value = part.partition(b"\r\n\r\n")
            value = value[:-2]  # the part's trailing CRLF
            name = head.split(b'name="')[1].split(b'"')[0].decode()
            if b"filename=" in head:
                with wave.open(io.BytesIO(value)) as w:
                    audio = (w.readframes(w.getnframes()), w.getframerate(), w.getnchannels())
            else:
                fields[name] = value.decode()
        self.uploads.append((fields, audio))
        return httpx.Response(200, json={"text": self.text})

    def _speak(self, request):
        body = json.loads(request.content)
        self.speech.append(body)
        pcm = struct.pack(f"<{240 * self.chunks}h", *([500] * 240 * self.chunks))  # 10 ms per chunk at 24 kHz
        if self.wav:
            header = io.BytesIO()
            with wave.open(header, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
                w.writeframes(b"")
            data = header.getvalue()[:40] + struct.pack("<I", len(pcm)) + pcm  # the 44-byte header, then samples
        else:
            data = pcm
        step_s = self.step_s

        async def stream():
            # odd-sized pieces: a sample may arrive split across two reads
            for i in range(0, len(data), 333):
                await asyncio.sleep(step_s)
                yield data[i:i + 333]

        return httpx.Response(200, headers={"content-type": "audio/pcm"}, content=stream())

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)


def stt(server, **options):
    rt = OpenAIHTTPSTT(ModelSpec(stage="stt", runtime="openai_http", model=URL,
                                 options={"model_name": "whisper", **options}))
    rt.transport = server.transport
    return rt


def tts(server, **options):
    rt = OpenAIHTTPTTS(ModelSpec(stage="tts", runtime="openai_http", model=URL,
                                 options={"model_name": "voice", "voice": "alloy", **options}))
    rt.transport = server.transport
    return rt


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("TEST_SPEECH_KEY", raising=False)


# ---- speech to text ----------------------------------------------------------------------------

async def test_stt_passes_the_conformance_kit():
    assert_conforms(await check_runtime(stt(FakeSpeechServer()), stt_audio=PCM))


async def test_a_window_goes_up_as_a_wav_file_with_the_model_and_language():
    server = FakeSpeechServer()
    rt = stt(server, language="en-US", extra_body={"temperature": 0})
    await rt.load()
    [result] = await rt.transcribe([STTRequest(audio=PCM, prompt="Order numbers: 1042")])
    assert isinstance(result, Transcript) and result.text == "Where is my order?"
    assert result.language == "en-US" and result.duration_s == 0.1  # as given; the server got "en"
    fields, (frames, rate, channels) = server.uploads[0]
    assert frames == PCM and rate == 16000 and channels == 1
    assert fields == {"model": "whisper", "response_format": "json", "language": "en",
                      "prompt": "Order numbers: 1042", "temperature": "0"}
    await rt.close()


async def test_windows_of_several_callers_go_up_together():
    server = FakeSpeechServer()
    rt = stt(server)
    await rt.load()
    results = await rt.transcribe([STTRequest(audio=PCM) for _ in range(3)])
    assert len(server.uploads) == 3 and all(r.text == "Where is my order?" for r in results)


async def test_stt_errors_land_in_the_request_slot():
    rt = stt(FakeSpeechServer(status=429))
    await rt.load()
    [limited, odd] = await rt.transcribe([STTRequest(audio=PCM), STTRequest(audio=b"\x01")])
    assert isinstance(limited, RateLimited) and limited.retry_after_s == 3.0
    assert isinstance(odd, InvalidRequest)


async def test_a_key_comes_from_the_environment_only(monkeypatch):
    server = FakeSpeechServer(key="sk-test")
    with pytest.raises(AuthFailed, match="TEST_SPEECH_KEY"):
        await stt(server, api_key_env="TEST_SPEECH_KEY").load()
    monkeypatch.setenv("TEST_SPEECH_KEY", "sk-test")
    rt = stt(server, api_key_env="TEST_SPEECH_KEY")
    await rt.load()
    assert (await rt.transcribe([STTRequest(audio=PCM)]))[0].text == "Where is my order?"


async def test_without_a_model_name_loading_says_what_to_set():
    rt = OpenAIHTTPSTT(ModelSpec(stage="stt", runtime="openai_http", model=URL))
    with pytest.raises(InvalidRequest, match="model_name"):
        await rt.load()


async def test_a_cancelled_window_stops_waiting_for_the_server():
    class Slow(FakeSpeechServer):
        async def handler(self, request):
            await asyncio.sleep(5)
            return httpx.Response(200, json={"text": "late"})

    server = Slow()
    rt = OpenAIHTTPSTT(ModelSpec(stage="stt", runtime="openai_http", model=URL,
                                 options={"model_name": "whisper", "verify": False}))
    rt.transport = httpx.MockTransport(server.handler)
    await rt.load()
    request = STTRequest(audio=PCM, cancel=CancelToken())
    asyncio.get_running_loop().call_later(0.05, request.cancel.cancel, "hung up")
    started = asyncio.get_running_loop().time()
    [result] = await rt.transcribe([request])
    assert isinstance(result, Cancelled) and asyncio.get_running_loop().time() - started < 1


# ---- text to speech ----------------------------------------------------------------------------

async def test_tts_passes_the_conformance_kit():
    assert_conforms(await check_runtime(tts(FakeSpeechServer(chunks=200, step_s=0.002))))


async def test_raw_pcm_streams_in_whole_samples_at_24_khz():
    server = FakeSpeechServer(chunks=10)
    rt = tts(server, extra_body={"instructions": "warm"})
    await rt.load()
    chunks = [c async for c in rt.synthesize(TTSRequest(text="Your order shipped.", speed=1.1))]
    assert len(chunks) > 1, "audio is passed on as it arrives, not after the whole sentence"
    assert all(len(c.pcm) % 2 == 0 and c.sample_rate == 24000 for c in chunks)
    assert sum(len(c.pcm) for c in chunks) == 240 * 10 * 2
    assert server.speech[0] == {"instructions": "warm", "model": "voice", "input": "Your order shipped.",
                                "voice": "alloy", "response_format": "pcm", "speed": 1.1}


async def test_a_wav_reply_is_read_for_its_rate_and_its_header_dropped():
    server = FakeSpeechServer(wav=True, chunks=10)
    rt = tts(server, response_format="wav")
    await rt.load()
    chunks = [c async for c in rt.synthesize(TTSRequest(text="Hello."))]
    assert {c.sample_rate for c in chunks} == {22050}
    assert b"".join(c.pcm for c in chunks) == struct.pack("<2400h", *([500] * 2400))


async def test_compressed_audio_is_refused_at_load():
    with pytest.raises(InvalidRequest, match="uncompressed"):
        await tts(FakeSpeechServer(), response_format="mp3").load()


async def test_tts_errors_are_raised_with_the_server_s_reason():
    rt = tts(FakeSpeechServer(status=400))
    await rt.load()
    with pytest.raises(InvalidRequest, match="nope"):
        [c async for c in rt.synthesize(TTSRequest(text="Hi."))]


async def test_a_server_that_sends_nothing_is_a_failure():
    rt = tts(FakeSpeechServer(chunks=0))
    await rt.load()
    with pytest.raises(RuntimeFailure, match="no audio"):
        [c async for c in rt.synthesize(TTSRequest(text="Hi."))]


# ---- choosing them -----------------------------------------------------------------------------

def test_a_url_as_the_speech_model_resolves_to_these_adapters():
    config = Agent(prompt="hi",
                   stt=STT("https://api.openai.com/v1", model_name="whisper-1", api_key_env="OPENAI_API_KEY"),
                   tts=TTS("http://localhost:8880/v1", model_name="kokoro", voice="af_heart")).config({})
    stt_spec = resolve_stage_config("stt", config.stt).spec
    tts_spec = resolve_stage_config("tts", config.tts).spec
    assert (stt_spec.runtime, stt_spec.model, stt_spec.options["model_name"]) == (
        "openai_http", "https://api.openai.com/v1", "whisper-1")
    assert (tts_spec.runtime, tts_spec.options["voice"]) == ("openai_http", "af_heart")


def test_a_misspelt_setting_is_refused():
    config = Agent(prompt="hi", tts=TTS("http://localhost:8880/v1", model_nam="kokoro")).config({})
    with pytest.raises(InvalidRequest, match="Did you mean 'model_name'"):
        resolve_stage_config("tts", config.tts)
