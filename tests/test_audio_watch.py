"""Noticing when the agent can't hear the caller: no frames, or only exact zeros."""
import json
import struct
import time

import pytest
from fusion_runtime import server
from fusion_runtime.engine.audio_watch import AudioWatch
from starlette.testclient import TestClient

ZEROS = b"\x00" * 640
SPEECH = struct.pack("<320h", *([900, -900] * 160))
ROOM = struct.pack("<320h", *([3, -2] * 160))  # a quiet room: tiny, never exactly zero


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_a_muted_mic_is_noticed_and_unmuting_clears_it():
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock)
    for _ in range(12):  # zeros from 0.25 s to 3.0 s: under 3 s of them so far
        clock.now += 0.25
        assert watch.frame(ZEROS) is None
    clock.now += 0.25
    problem = watch.frame(ZEROS)  # 3 s of nothing but zeros
    assert problem["problem"] == "silent_audio" and problem["seconds"] == 3.0 and "muted" in problem["message"]
    clock.now += 0.25
    assert watch.frame(ZEROS) is None  # said once per episode, not every frame
    clock.now += 2.0
    ok = watch.frame(SPEECH)
    assert ok["event"] == "audio.ok" and ok["problem"] == "silent_audio" and ok["lasted_s"] >= 2.0


def test_a_quiet_caller_is_not_a_dead_one():
    """Thinking in a quiet room still sends room noise, never exactly zero."""
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock)
    for _ in range(100):
        clock.now += 0.2
        assert watch.frame(ROOM) is None and watch.check() is None


def test_audio_that_stops_arriving_is_noticed_by_the_timer():
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock)
    watch.frame(SPEECH)
    clock.now += 2.9
    assert watch.check() is None
    clock.now += 0.2
    problem = watch.check()
    assert problem["problem"] == "no_audio" and problem["seconds"] == 3.1
    assert watch.check() is None
    clock.now += 1.0
    assert watch.frame(SPEECH)["event"] == "audio.ok"


def test_off_means_off():
    watch = AudioWatch(after_s=0)
    assert watch.frame(ZEROS) is None and watch.check() is None


class EchoOrchestrator:
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
    monkeypatch.setenv("FUSION_DEAD_AUDIO_S", "0.5")
    monkeypatch.setattr(server, "PipelineOrchestrator", EchoOrchestrator)
    with TestClient(server.app, client=("127.0.0.1", 4000)) as client:
        yield client


def next_text(ws, wanted, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        message = ws.receive()
        if message.get("text"):
            event = json.loads(message["text"])
            if event.get("type") == wanted:
                return event
    raise AssertionError(f"no {wanted} message")


def test_the_caller_is_told_when_their_mic_sends_only_silence(local):
    with local.websocket_connect("/v1/voice/ws") as ws:
        assert ws.receive_json()["type"] == "config"
        for _ in range(15):  # 0.75 s of a muted mic
            ws.send_bytes(ZEROS)
            time.sleep(0.05)
        problem = next_text(ws, "audio_problem")
        assert problem["problem"] == "silent_audio" and "muted" in problem["message"]
        ws.send_bytes(SPEECH)
        assert next_text(ws, "audio_ok") == {"type": "audio_ok"}


# ---- phone lines: a dead one is hiss, not zeros ------------------------------------------------

HISS = struct.pack("<320h", *([25, -25] * 160))  # RMS 25: under Asif Ali's ~40 for a dead path


def test_a_phone_line_with_only_hiss_is_reported_as_quiet():
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock, floor_rms=40, quiet_after_s=10.0)
    for _ in range(40):  # 8 s of hiss: under 10 s so far, and never "muted": it isn't zeros
        clock.now += 0.2
        assert watch.frame(HISS) is None
    for _ in range(15):
        clock.now += 0.2
        problem = watch.frame(HISS)
        if problem:
            break
    assert problem["problem"] == "quiet_line" and "dead phone line" in problem["message"]
    assert not watch.heard_caller and round(watch.peak_rms) == 25
    clock.now += 0.2
    ok = watch.frame(SPEECH)  # the caller speaks: it was a quiet caller after all
    assert ok["event"] == "audio.ok" and ok["problem"] == "quiet_line" and watch.heard_caller


def test_without_a_floor_hiss_is_just_a_quiet_caller():
    """The browser keeps today's checks: no floor, so a quiet room is never a problem."""
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock)
    for _ in range(100):
        clock.now += 0.2
        assert watch.frame(HISS) is None
    assert round(watch.peak_rms) == 25  # the loudest moment is still kept


def test_audio_coming_back_quietly_ends_no_audio():
    clock = Clock()
    watch = AudioWatch(after_s=3.0, clock=clock, floor_rms=40, quiet_after_s=10.0)
    clock.now += 3.5
    assert watch.check()["problem"] == "no_audio"
    clock.now += 0.2
    assert watch.frame(HISS)["event"] == "audio.ok"  # frames again, even if only the line's hiss
