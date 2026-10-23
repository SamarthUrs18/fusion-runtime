"""Plivo's half of a phone call, for testing fusion without Plivo or a number.

1. POST the number's answer URL (fusion's /v1/phone/plivo) with the call's details
2. read the XML reply: a bidirectional <Stream> and its URL
3. open the stream, signed the way Plivo signs it: X-Plivo-Signature-V3 = base64(HMAC-SHA256 of
   "GET" + the URL + X-Plivo-Signature-V3-Nonce, with the account's auth token)
4. `start`, the caller as `media` (8 kHz μ-law, 20 ms each); fusion's `playAudio` is played in order,
   each `checkpoint` answered with `playedStream` once the audio before it has played, `clearAudio`
   answered with `clearedAudio`. Hanging up closes the socket.

Written from Plivo's documentation (plivo.com/docs/voice-agents/audio-streaming), not fusion's code.
"""
import base64
import hashlib
import hmac
import json
import secrets
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from phone_call import (  # noqa: F401 - Playback is part of this module's API
    PHONE_RATE,
    Call,
    Playback,
)

FRAME_BYTES = 160  # 20 ms of 8 kHz μ-law


def sign(auth_token: str, url: str, nonce: str) -> str:
    digest = hmac.new(auth_token.encode(), ("GET" + url + nonce).encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def parse_stream_xml(xml: str):
    """(stream URL or None, its attributes, what <Speak> said)."""
    root = ET.fromstring(xml)
    stream = root.find("Stream")
    said = [element.text or "" for element in root.iter("Speak")]
    if stream is None:
        return None, {}, said
    return (stream.text or "").strip(), dict(stream.attrib), said


def start(stream_id: str, call_id: str) -> str:
    return json.dumps({"event": "start", "sequenceNumber": 1,
                       "start": {"callId": call_id, "streamId": stream_id, "accountId": "MA" + "0" * 18,
                                 "tracks": ["inbound"], "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}},
                       "extra_headers": ""})


def media(stream_id: str, mulaw_frame: bytes, sequence: int, chunk: int, timestamp_ms: int) -> str:
    return json.dumps({"event": "media", "sequenceNumber": sequence, "streamId": stream_id,
                       "media": {"track": "inbound", "timestamp": str(timestamp_ms), "chunk": chunk,
                                 "payload": base64.b64encode(mulaw_frame).decode()},
                       "extra_headers": ""})


def played(stream_id: str, name: str, sequence: int) -> str:
    return json.dumps({"event": "playedStream", "sequenceNumber": sequence, "streamId": stream_id, "name": name})


def cleared(stream_id: str, sequence: int) -> str:
    return json.dumps({"event": "clearedAudio", "sequenceNumber": sequence, "streamId": stream_id})


class PlivoCall(Call):
    provider = "plivo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = 0

    async def send_caller_audio(self, samples: np.ndarray) -> None:
        from fusion_runtime.phone import mulaw as codec

        data = codec.encode(samples)
        for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
            self.chunk += 1
            await self.ws.send(media(self.stream_id, data[i:i + FRAME_BYTES], self.next_sequence(), self.chunk,
                                     self.chunk * 20))

    def read(self, message: dict):
        from fusion_runtime.phone import mulaw as codec

        event = message.get("event")
        if event == "playAudio":
            m = message.get("media") or {}
            if m.get("contentType") != "audio/x-mulaw" or int(m.get("sampleRate", 0)) != PHONE_RATE:
                self.on_note(f"playAudio doesn't match the stream's format: {m.get('contentType')} {m.get('sampleRate')}")
            return "audio", codec.decode(base64.b64decode(m.get("payload") or "")).tobytes()
        if event == "checkpoint":
            return "marker", message.get("name")
        if event == "clearAudio":
            return "clear", None
        return "other", event

    async def acknowledge(self, name: str) -> None:
        await self.ws.send(played(self.stream_id, name, self.next_sequence()))

    async def cleared(self, pending) -> None:
        await self.ws.send(cleared(self.stream_id, self.next_sequence()))  # dropped checkpoints aren't reported


async def place_call(base_url: str, auth_token: str, playback: Playback, caller: str = "+15550100001",
                     called: str = "+15550100999", **call_options):
    import httpx
    import websockets

    call_id = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=10) as http:
        response = await http.post(base_url.rstrip("/") + "/v1/phone/plivo",
                                   data={"CallUUID": call_id, "From": caller, "To": called, "Direction": "inbound"})
    if response.status_code != 200:
        raise RuntimeError(f"fusion answered the call's answer URL with {response.status_code}: {response.text}")
    url, attributes, said = parse_stream_xml(response.text)
    if not url:
        raise RuntimeError(f"fusion didn't stream the call; it said: {' '.join(said) or response.text}")
    if attributes.get("bidirectional") != "true":
        raise RuntimeError(f"fusion's <Stream> isn't bidirectional: {attributes}")
    nonce = secrets.token_hex(8)
    headers = {"X-Plivo-Signature-V3": sign(auth_token, url, nonce), "X-Plivo-Signature-V3-Nonce": nonce}
    try:
        ws = await websockets.connect(url, additional_headers=headers)
    except TypeError:  # websockets before 14
        ws = await websockets.connect(url, extra_headers=headers)
    stream_id = str(uuid.uuid4())
    await ws.send(start(stream_id, call_id))
    return PlivoCall(ws, stream_id, call_id, playback, **call_options)
