"""Exotel (India): answer an Exotel number with a fusion agent, through its Voicebot applet.

    caller ──phone──► Exotel ══ WebSocket /v1/phone/exotel/stream ══► fusion   the call's audio

There's no webhook: in the call flow, the Voicebot applet is given fusion's stream URL directly,
with one of fusion's keys as Basic auth, which Exotel sends as an Authorization header:

    wss://fusion:<a fusion key>@your-server/v1/phone/exotel/stream?sample-rate=16000

Then the same shape of conversation as Twilio's, in Exotel's words: `connected`, `start` (with
the call's ids and `media_format`), `media` carrying raw 16-bit little-endian PCM (8 kHz unless
`sample-rate` asks for 16000 or 24000), `mark` echoed once audio has been processed, `clear`
to drop what hasn't started playing, `stop` when the call ends. Closing the socket moves the call
to the next applet.

Exotel's pages disagree on some details (numbers sent as numbers or strings), so both are taken.
Audio sent back has to come in chunks of at least 3200 bytes, at most 100 KB, and a multiple of
320 bytes, or there are gaps: 100 ms chunks (200 ms at 8 kHz, to stay over the minimum), the last
of a reply padded with silence to a whole chunk.

The protocol is Exotel's, documented at developer.exotel.com/docs/agentstream; this is our own
implementation of it.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fusion_runtime.phone.link import PCM16, NotThisProvider, PhoneLink, number

STREAM_PATH = "/v1/phone/exotel/stream"
RATES = (8000, 16000, 24000)
START_TIMEOUT_S = 10.0


@dataclass
class Start:
    stream_sid: str
    call_sid: str
    sample_rate: int = 8000
    parameters: Dict[str, str] = field(default_factory=dict)


async def read_start(websocket, timeout_s: float = START_TIMEOUT_S) -> Start:
    """Exotel's first messages: `connected`, then `start`."""
    async def read():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise NotThisProvider("the stream closed before it started")
            try:
                data = json.loads(message.get("text") or "")
            except ValueError:
                raise NotThisProvider("the stream sent something other than Exotel's JSON messages") from None
            event = data.get("event")
            if event == "connected":
                continue
            if event != "start":
                raise NotThisProvider(f"expected Exotel's start message, got {event!r}")
            start = data.get("start") or {}
            rate = number((start.get("media_format") or {}).get("sample_rate"), 8000)
            if rate not in RATES:
                raise NotThisProvider(f"Exotel streams at {', '.join(map(str, RATES))} Hz, not {rate}")
            return Start(stream_sid=start.get("stream_sid") or data.get("stream_sid") or "",
                         call_sid=start.get("call_sid") or "", sample_rate=rate,
                         parameters={str(k): str(v) for k, v in (start.get("custom_parameters") or {}).items()})

    try:
        return await asyncio.wait_for(read(), timeout_s)
    except asyncio.TimeoutError:
        raise NotThisProvider(f"no start message within {timeout_s:.0f} s") from None


def basic_password(authorization: Optional[str]) -> Optional[str]:
    """The password of an `Authorization: Basic ...` header: where Exotel puts the URL's credentials."""
    import base64
    import binascii

    if not authorization or not authorization.lower().startswith("basic "):
        return None
    try:
        decoded = base64.b64decode(authorization[6:].strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    return decoded.partition(":")[2] or None


class ExotelLink(PhoneLink):
    """A phone call on Exotel's Voicebot applet: 16-bit PCM at the stream's rate, marks, clear."""

    kind = "exotel"

    def __init__(self, websocket, start: Start, pipeline_rate: int, reply_rate: int):
        chunk_ms = 200 if start.sample_rate == 8000 else 100  # never under Exotel's 3200-byte minimum
        super().__init__(websocket, start.call_sid, start.stream_sid, line_rate=start.sample_rate, encoding=PCM16,
                         pipeline_rate=pipeline_rate, reply_rate=reply_rate, frame_ms=chunk_ms)
        self.start = start

    def parse(self, data: Dict[str, Any]):
        event = data.get("event")
        if event == "media":
            return "media", (data.get("media") or {}).get("payload")
        if event == "mark":
            return "played", (data.get("mark") or {}).get("name")
        if event == "stop":
            return "stop", None
        if event == "dtmf":
            return "dtmf", (data.get("dtmf") or {}).get("digit")
        return "other", None

    def media_message(self, payload_b64: str) -> Dict[str, Any]:
        return {"event": "media", "stream_sid": self.stream_id, "media": {"payload": payload_b64}}

    def marker_message(self, name: str) -> Dict[str, Any]:
        return {"event": "mark", "stream_sid": self.stream_id, "mark": {"name": name}}

    def clear_message(self) -> Dict[str, Any]:
        return {"event": "clear", "stream_sid": self.stream_id}
