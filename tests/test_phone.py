"""Phone calls through Twilio: μ-law, the signed webhook, the stream, and a call end to end.

The Twilio side is the phone simulator's (tools/phone_sim/twilio_side.py), written separately
from fusion's from Twilio's documentation, so the two have to agree for a call to work.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fusion_runtime import server
from fusion_runtime.phone import mulaw, twilio
from fusion_runtime.telemetry import ListSink, telemetry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "phone_sim"))
import twilio_side  # noqa: E402

TOKEN = "test-twilio-auth-token"
WEBHOOK = "http://testserver/v1/phone/twilio"


# ---- μ-law -------------------------------------------------------------------------------------

def test_mulaw_matches_the_reference_codec_for_every_value():
    audioop = pytest.importorskip("audioop")  # gone in Python 3.13; the check runs on 3.11 and 3.12
    every = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    assert mulaw.encode(every) == audioop.lin2ulaw(every.tobytes(), 2)
    codes = bytes(range(256))
    assert np.array_equal(mulaw.decode(codes), np.frombuffer(audioop.ulaw2lin(codes, 2), dtype=np.int16))


def test_mulaw_round_trip_keeps_speech_close_and_silence_silent():
    speech = (np.sin(np.linspace(0, 200, 8000)) * 12000).astype(np.int16)
    back = mulaw.decode(mulaw.encode(speech)).astype(np.int32)
    assert np.max(np.abs(back - speech)) < 12000 * 0.04  # μ-law's few percent, at this level
    assert mulaw.decode(mulaw.SILENCE)[0] == 0


# ---- the webhook -------------------------------------------------------------------------------

def test_fusion_and_twilio_sign_the_same_way():
    params = twilio_side.call_parameters("CA1", "+15550100001", "+15550100999")
    url = "https://voice.example.com/v1/phone/twilio"
    assert twilio.signature(TOKEN, url, params) == twilio_side.sign(TOKEN, url, params)
    good = twilio_side.sign(TOKEN, url, params)
    assert twilio.valid_signature(TOKEN, url, params, good)
    # signed as configured, received through a proxy that kept the default port
    assert twilio.valid_signature(TOKEN, "https://voice.example.com:443/v1/phone/twilio", params, good)
    assert not twilio.valid_signature(TOKEN, url, {**params, "From": "+15550100002"}, good)  # tampered
    assert not twilio.valid_signature("another-account", url, params, good)
    assert not twilio.valid_signature(TOKEN, url, params, None)


def test_the_reply_connects_the_call_to_the_stream_with_its_token():
    xml = twilio.connect_twiml("wss://voice.example.com/v1/phone/twilio/stream", {"token": 'a"b<c', "call_sid": "CA1"})
    url, params, said = twilio_side.parse_twiml(xml)
    assert url == "wss://voice.example.com/v1/phone/twilio/stream" and params == {"token": 'a"b<c', "call_sid": "CA1"}
    assert twilio.stream_url("https://voice.example.com/v1/phone/twilio") == "wss://voice.example.com/v1/phone/twilio/stream"
    assert twilio.stream_url("http://127.0.0.1:8000/v1/phone/twilio") == "ws://127.0.0.1:8000/v1/phone/twilio/stream"
    assert twilio_side.parse_twiml(twilio.refuse_twiml("All lines are busy."))[2] == ["All lines are busy."]


# ---- a call, end to end ------------------------------------------------------------------------

class FakeOrchestrator:
    """Answers each chunk of caller audio with a little audio of its own, like a reply."""

    def __init__(self, config):
        self.config = config
        self.ready = True

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def run_pipeline(self, audio_stream, system_prompt, on_event=None, barge_in=None, trace=None, tools=()):
        heard = 0
        async for chunk in audio_stream:
            heard += 1
            FakeOrchestrator.heard.append(chunk)
            if heard == 4:
                on_event({"type": "interrupted"})  # as if the caller had talked over the reply
            yield (np.ones(480, dtype=np.int16) * 3000).tobytes()  # 20 ms at 24 kHz


@pytest.fixture
def phone(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")
    monkeypatch.setenv(twilio.AUTH_TOKEN_ENV, TOKEN)
    monkeypatch.delenv("FUSION_PUBLIC_URL", raising=False)
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    FakeOrchestrator.heard = []
    sink = ListSink()
    with TestClient(server.app, client=("203.0.113.7", 40000)) as client:  # from the internet, like Twilio
        telemetry.add_sink(sink)
        try:
            yield client, sink
        finally:
            telemetry.remove_sink(sink)


def ring(client, params=None, signature=None):
    params = params or twilio_side.call_parameters("CA" + "1" * 32, "+15550100001", "+15550100999")
    return client.post("/v1/phone/twilio", data=params,
                       headers={"X-Twilio-Signature": signature or twilio_side.sign(TOKEN, WEBHOOK, params)})


def test_a_call_is_answered_heard_and_spoken_to(phone):
    client, sink = phone
    reply = ring(client)
    assert reply.status_code == 200 and reply.headers["content-type"].startswith("text/xml")
    url, params, _ = twilio_side.parse_twiml(reply.text)
    assert url == "ws://testserver/v1/phone/twilio/stream" and params["token"]

    noise = mulaw.encode(np.random.default_rng(1).normal(0, 2000, 160 * 12))
    with client.websocket_connect("/v1/phone/twilio/stream") as ws:
        ws.send_text(twilio_side.connected())
        ws.send_text(twilio_side.start("MZ1", "CA" + "1" * 32, params))
        for i in range(12):
            ws.send_text(twilio_side.media("MZ1", noise[i * 160:(i + 1) * 160], i + 3, i + 1, (i + 1) * 20))
        sent = [json.loads(ws.receive_text()) for _ in range(8)]
        ws.send_text(twilio_side.stop("MZ1", "CA" + "1" * 32, 99))

    # the caller's 8 kHz μ-law reached the pipeline as 16 kHz PCM: 20 ms is 640 bytes
    assert FakeOrchestrator.heard and all(len(chunk) % 2 == 0 for chunk in FakeOrchestrator.heard)
    assert sum(len(c) for c in FakeOrchestrator.heard[2:]) == pytest.approx(640 * (len(FakeOrchestrator.heard) - 2), rel=0.05)
    # the agent's 24 kHz audio went back as 20 ms frames of 8 kHz μ-law, each reply followed by a mark
    media = [m for m in sent if m["event"] == "media"]
    assert media and all(len(base64.b64decode(m["media"]["payload"])) == 160 for m in media)
    assert all(m["streamSid"] == "MZ1" for m in sent)
    assert any(m["event"] == "mark" for m in sent)
    # the interruption cleared what Twilio hadn't played yet
    assert any(m["event"] == "clear" for m in sent)

    start = sink.named("session.start")[0]
    assert start.attrs["transport"] == "twilio" and start.attrs["call_sid"] == "CA" + "1" * 32
    assert sink.named("session.end")[0].attrs["reason"] == "client_disconnected"


def test_a_webhook_not_signed_by_twilio_is_refused(phone):
    client, sink = phone
    reply = ring(client, signature="forged")
    assert reply.status_code == 403 and "Stream" not in reply.text
    assert sink.named("auth.failed")


def test_without_the_auth_token_phone_calls_are_off(phone, monkeypatch):
    client, _ = phone
    monkeypatch.delenv(twilio.AUTH_TOKEN_ENV)
    reply = ring(client)
    assert reply.status_code == 503 and twilio.AUTH_TOKEN_ENV in reply.json()["error"]["fix"]


def test_a_stream_without_a_valid_token_is_closed(phone):
    client, sink = phone
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/v1/phone/twilio/stream") as ws:
        ws.send_text(twilio_side.connected())
        ws.send_text(twilio_side.start("MZ1", "CA1", {"token": "made-up"}))
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1008 and sink.named("auth.failed")


def test_a_token_works_once(phone):
    client, _ = phone
    from starlette.websockets import WebSocketDisconnect

    _, params, _ = twilio_side.parse_twiml(ring(client).text)
    for attempt in range(2):
        with client.websocket_connect("/v1/phone/twilio/stream") as ws:
            ws.send_text(twilio_side.connected())
            ws.send_text(twilio_side.start("MZ1", "CA1", params))
            if attempt == 0:
                ws.send_text(twilio_side.stop("MZ1", "CA1", 3))
            else:
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_text()
                assert closed.value.code == 1008


def test_a_stream_that_isnt_twilio_is_closed(phone):
    client, _ = phone
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/v1/phone/twilio/stream") as ws:
        ws.send_bytes(b"\x00\x01" * 160)  # raw PCM, as a browser would send
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1008


# ---- what the caller hears, and when the agent may be interrupted ------------------------------

class _Socket:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


async def test_playback_follows_twilios_marks():
    """Interruption needs to know the caller is still hearing the agent: until Twilio echoes the
    last mark, it is."""
    link = twilio.TwilioLink(_Socket(), twilio.Start("MZ1", "CA1"), pipeline_rate=16000, reply_rate=24000)
    states = []
    link.controls = lambda message: states.append(message["playing"])
    await link.send_audio((np.ones(2400, dtype=np.int16) * 3000).tobytes())  # 100 ms of reply
    await link.send_audio((np.ones(2400, dtype=np.int16) * 3000).tobytes())
    assert states == [True]
    marks = [m for m in link.websocket.sent if m["event"] == "mark"]
    assert len(marks) == 2

    async def twilio_says(*events):
        queue = [{"type": "websocket.receive", "text": e} for e in events]

        async def receive():
            return queue.pop(0) if queue else {"type": "websocket.disconnect", "code": 1000}
        link.websocket.receive = receive
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            while True:
                await link.receive()

    await twilio_says(twilio_side.mark("MZ1", marks[0]["mark"]["name"], 5))
    assert states == [True]  # one reply still playing
    await twilio_says(twilio_side.mark("MZ1", marks[1]["mark"]["name"], 6))
    assert states == [True, False]
