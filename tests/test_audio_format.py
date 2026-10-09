"""The caller's audio is checked before it's used: aligned samples, no WAV header, no compressed audio."""
import json
import struct

import pytest
from fusion_runtime import server
from fusion_runtime.engine.audio_format import BadAudioFormat, Intake
from starlette.testclient import TestClient


def wav(rate=16000, channels=1, bits=16, encoding=1, samples=b"\x01\x02" * 160):
    fmt = struct.pack("<HHIIHH", encoding, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_an_odd_byte_is_carried_so_later_samples_stay_aligned():
    intake = Intake()
    assert intake.accept(b"\x01\x02\x03") == b"\x01\x02"
    assert intake.accept(b"\x04\x05\x06") == b"\x03\x04\x05\x06"  # the stray byte rejoins its sample
    assert intake.accept(b"\x07\x08") == b"\x07\x08"


def test_a_wav_header_in_the_right_format_is_removed_once():
    intake = Intake(16000)
    first = intake.accept(wav())
    assert first == b"\x01\x02" * 160 and "WAV header was removed" in intake.note
    intake.note = None
    assert intake.accept(b"RIFF") == b"RIFF"  # only the first message is checked; later ones are audio


@pytest.mark.parametrize("header", [wav(rate=44100), wav(channels=2), wav(bits=8), wav(encoding=3)])
def test_a_wav_in_another_format_is_named_with_a_fix(header):
    with pytest.raises(BadAudioFormat, match=r"WAV file with .*ffmpeg -i in.wav -ac 1 -ar 16000"):
        Intake(16000).accept(header)


@pytest.mark.parametrize("start, name", [(b"ID3\x04", "MP3"), (b"OggS\x00", "Ogg"),
                                         (b"\x1a\x45\xdf\xa3", "WebM"), (b"fLaC", "FLAC")])
def test_compressed_audio_is_refused_by_name(start, name):
    with pytest.raises(BadAudioFormat, match=rf"this is {name}.*send raw 16-bit little-endian mono PCM"):
        Intake().accept(start + b"\x00" * 100)


def test_plain_pcm_passes_untouched():
    pcm = struct.pack("<320h", *range(320))
    assert Intake().accept(pcm) == pcm
    # PCM that happens to start like an untagged MP3 frame (FF FB) is still PCM
    lookalike = b"\xff\xfb" + b"\x10\x00" * 319
    assert Intake().accept(lookalike) == lookalike


class QuietOrchestrator:
    def __init__(self, config):
        self.config, self.ready = config, True

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def run_pipeline(self, audio_stream, system_prompt, on_event=None, barge_in=None, trace=None, tools=()):
        async for _chunk in audio_stream:
            if False:
                yield b""


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")
    monkeypatch.setattr(server, "PipelineOrchestrator", QuietOrchestrator)
    with TestClient(server.app, client=("127.0.0.1", 4000)) as client:
        yield client


def test_mp3_over_the_socket_ends_the_call_with_the_reason(local):
    with local.websocket_connect("/v1/voice/ws") as ws:
        assert ws.receive_json()["type"] == "config"
        ws.send_bytes(b"ID3\x04" + b"\x00" * 600)
        error = ws.receive_json()
        assert error["code"] == "bad_audio_format" and "this is MP3" in error["message"] and not error["retryable"]


def test_a_wav_header_is_removed_and_the_caller_is_told(local):
    with local.websocket_connect("/v1/voice/ws") as ws:
        assert ws.receive_json()["type"] == "config"
        ws.send_bytes(wav())
        message = json.loads(ws.receive_text())
        assert message["type"] == "warning" and message["code"] == "wav_header"
