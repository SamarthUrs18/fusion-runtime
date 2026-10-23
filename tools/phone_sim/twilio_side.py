"""Twilio's half of a phone call, for testing fusion without Twilio or a phone number.

Does what Twilio does when someone calls a number whose voice URL points at fusion:

    1. POST the webhook, signed with the account's auth token (X-Twilio-Signature)
    2. read the TwiML reply: the stream URL and its parameters
    3. open the stream: `connected`, `start`, then the caller's audio as `media` (20 ms of
       8 kHz μ-law each), and `stop` on hang-up
    4. play what fusion sends back, in order; echo each `mark` once the audio before it has
       played; on `clear`, drop what hasn't played

The signing and the message formats are written here from Twilio's documentation, not imported
from fusion, so a mistake on fusion's side shows up as a failed call instead of agreeing with
itself. μ-law comes from fusion (it's checked against the reference encoder in fusion's tests).
"""
import base64
import hashlib
import hmac
import json
import uuid
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
from phone_call import (  # noqa: F401 - Playback is part of this module's API
    PHONE_RATE,
    Call,
    Playback,
)

FRAME_MS = 20
FRAME_BYTES = PHONE_RATE * FRAME_MS // 1000  # one byte a sample in μ-law


def sign(auth_token: str, url: str, params: Dict[str, str]) -> str:
    """Twilio's request signature: HMAC-SHA1 over the URL plus each parameter name and value,
    sorted by name, base64-encoded."""
    data = url + "".join(name + params[name] for name in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def call_parameters(call_sid: str, caller: str, called: str) -> Dict[str, str]:
    """The form fields Twilio posts for an incoming call (the ones that matter here)."""
    return {"AccountSid": "AC" + "0" * 32, "CallSid": call_sid, "CallStatus": "ringing", "Direction": "inbound",
            "From": caller, "To": called, "Caller": caller, "Called": called, "ApiVersion": "2010-04-01"}


def parse_twiml(xml: str) -> Tuple[Optional[str], Dict[str, str], List[str]]:
    """(stream URL or None, its parameters, what <Say> said) from a TwiML response."""
    root = ET.fromstring(xml)
    stream = root.find("./Connect/Stream")
    said = [element.text or "" for element in root.iter("Say")]
    if stream is None:
        return None, {}, said
    params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
    return stream.get("url"), params, said


# ---- messages Twilio sends on the stream ------------------------------------------------------

def connected() -> str:
    return json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})


def start(stream_sid: str, call_sid: str, parameters: Dict[str, str], sequence: int = 1) -> str:
    return json.dumps({
        "event": "start", "sequenceNumber": str(sequence), "streamSid": stream_sid,
        "start": {"accountSid": "AC" + "0" * 32, "streamSid": stream_sid, "callSid": call_sid,
                  "tracks": ["inbound"], "customParameters": parameters,
                  "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": PHONE_RATE, "channels": 1}},
    })


def media(stream_sid: str, mulaw_frame: bytes, sequence: int, chunk: int, timestamp_ms: int) -> str:
    return json.dumps({"event": "media", "sequenceNumber": str(sequence), "streamSid": stream_sid,
                       "media": {"track": "inbound", "chunk": str(chunk), "timestamp": str(timestamp_ms),
                                 "payload": base64.b64encode(mulaw_frame).decode()}})


def mark(stream_sid: str, name: str, sequence: int) -> str:
    return json.dumps({"event": "mark", "sequenceNumber": str(sequence), "streamSid": stream_sid,
                       "mark": {"name": name}})


def stop(stream_sid: str, call_sid: str, sequence: int) -> str:
    return json.dumps({"event": "stop", "sequenceNumber": str(sequence), "streamSid": stream_sid,
                       "stop": {"accountSid": "AC" + "0" * 32, "callSid": call_sid}})


# ---- the call ----------------------------------------------------------------------------------

class TwilioCall(Call):
    provider = "twilio"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = 0

    async def send_caller_audio(self, samples: np.ndarray) -> None:
        from fusion_runtime.phone import mulaw as codec

        data = codec.encode(samples)
        for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
            self.chunk += 1
            await self.ws.send(media(self.stream_id, data[i:i + FRAME_BYTES], self.next_sequence(), self.chunk,
                                     self.chunk * FRAME_MS))

    def read(self, message: dict):
        from fusion_runtime.phone import mulaw as codec

        event = message.get("event")
        if event == "media":
            return "audio", codec.decode(base64.b64decode(message["media"]["payload"])).tobytes()
        if event == "mark":
            return "marker", message["mark"]["name"]
        if event == "clear":
            return "clear", None
        return "other", event

    async def acknowledge(self, name: str) -> None:
        await self.ws.send(mark(self.stream_id, name, self.next_sequence()))

    async def hang_up(self) -> None:
        await self.ws.send(stop(self.stream_id, self.call_id, self.next_sequence()))
        await self.ws.close()


# Kept for the tests and the simulator: the playback model and the call, under their old names
Call = TwilioCall


async def place_call(base_url: str, auth_token: str, playback: Playback, caller: str = "+15550100001",
                     called: str = "+15550100999", **call_options):
    """Ring fusion: the signed webhook, then the stream it asks for. Returns the open call, or raises
    with what fusion answered instead."""
    import httpx
    import websockets

    call_sid = "CA" + uuid.uuid4().hex
    params = call_parameters(call_sid, caller, called)
    url = base_url.rstrip("/") + "/v1/phone/twilio"
    async with httpx.AsyncClient(timeout=10) as http:
        response = await http.post(url, data=params, headers={"X-Twilio-Signature": sign(auth_token, url, params)})
    if response.status_code != 200:
        raise RuntimeError(f"fusion answered the call's webhook with {response.status_code}: {response.text}")
    stream_url, stream_params, said = parse_twiml(response.text)
    if stream_url is None:
        raise RuntimeError(f"fusion didn't connect the call; it said: {' '.join(said) or response.text}")
    ws = await websockets.connect(stream_url)
    stream_sid = "MZ" + uuid.uuid4().hex
    await ws.send(connected())
    await ws.send(start(stream_sid, call_sid, stream_params))
    return TwilioCall(ws, stream_sid, call_sid, playback, **call_options)
