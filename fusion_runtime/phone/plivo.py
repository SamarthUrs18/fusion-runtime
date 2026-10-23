"""Plivo: answer a Plivo number with a fusion agent, on the customer's own Plivo account.

    caller ──phone──► Plivo ──POST /v1/phone/plivo──► fusion        "stream this call to me"
                      Plivo ◄═══ WebSocket /v1/phone/plivo/stream ═══► fusion   the call's audio

1. The number's answer URL asks fusion what to do; fusion answers with Plivo XML: a bidirectional
   <Stream> of 8 kHz μ-law to its stream endpoint. The answer carries no secret, so it needs none.
2. Plivo signs the stream's WebSocket connection with the account's auth token
   (X-Plivo-Signature-V3: HMAC-SHA256 of "GET" + the full URL + X-Plivo-Signature-V3-Nonce,
   base64). fusion checks that, with FUSION_PLIVO_AUTH_TOKEN, before taking the call.
3. Then: `start` (call and stream ids, media format), `media` from the caller, `playAudio` back,
   `checkpoint` after each reply, answered by `playedStream` once it has played, `clearAudio` when
   the caller interrupts. The call ends when the socket closes.

The protocol is Plivo's, documented at plivo.com/docs/voice-agents/audio-streaming; this is our own
implementation of it.
"""
import asyncio
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit
from xml.sax.saxutils import escape, quoteattr

from fusion_runtime.phone.link import MULAW, NotThisProvider, PhoneLink
from fusion_runtime.security.keys import Principal

AUTH_TOKEN_ENV = "FUSION_PLIVO_AUTH_TOKEN"
ANSWER_PATH = "/v1/phone/plivo"
STREAM_PATH = "/v1/phone/plivo/stream"
PHONE_RATE = 8000
CONTENT_TYPE = "audio/x-mulaw"
START_TIMEOUT_S = 10.0

PRINCIPAL = Principal(fingerprint="plivo", name="plivo", via="plivo")


def signature(auth_token: str, url: str, nonce: str) -> str:
    """X-Plivo-Signature-V3 for a WebSocket connection: HMAC-SHA256 of GET + URL + nonce, base64."""
    digest = hmac.new(auth_token.encode(), f"GET{url}{nonce}".encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def valid_signature(auth_token: str, urls: Iterable[str], nonce: Optional[str], given: Optional[str]) -> bool:
    """Whether Plivo opened this stream. Plivo's examples build the URL from the connection's own
    scheme and Host, so both ws(s) and http(s) spellings are tried; a header holding several
    signatures (comma separated, as when a token is rotated) matches on any one."""
    if not auth_token or not nonce or not given:
        return False
    candidates = [g.strip() for g in given.split(",") if g.strip()]
    for url in urls:
        for spelling in _spellings(url):
            expected = signature(auth_token, spelling, nonce)
            if any(hmac.compare_digest(expected, g) for g in candidates):
                return True
    return False


def _spellings(url: str):
    parts = urlsplit(url)
    swap = {"wss": "https", "https": "wss", "ws": "http", "http": "ws"}
    yield url
    if parts.scheme in swap:
        yield urlunsplit((swap[parts.scheme], parts.netloc, parts.path, parts.query, parts.fragment))


def stream_url(answer_url: str) -> str:
    parts = urlsplit(answer_url)
    return urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc, STREAM_PATH, "", ""))


def stream_xml(url: str) -> str:
    """Plivo XML that streams the call both ways; the call ends when the stream does."""
    return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
            f'<Stream bidirectional="true" keepCallAlive="true" contentType={quoteattr(f"{CONTENT_TYPE};rate={PHONE_RATE}")}>'
            f"{escape(url)}</Stream></Response>")


def refuse_xml(message: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Speak>{escape(message)}</Speak><Hangup/></Response>'


@dataclass
class Start:
    stream_id: str
    call_id: str


async def read_start(websocket, timeout_s: float = START_TIMEOUT_S) -> Start:
    async def read():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise NotThisProvider("the stream closed before it started")
            try:
                data = json.loads(message.get("text") or "")
            except ValueError:
                raise NotThisProvider("the stream sent something other than Plivo's JSON messages") from None
            event = data.get("event")
            if event != "start":
                raise NotThisProvider(f"expected Plivo's start event, got {event!r}")
            start = data.get("start") or {}
            media = start.get("mediaFormat") or {}
            if media and (media.get("encoding") != CONTENT_TYPE or int(media.get("sampleRate", 0)) != PHONE_RATE):
                raise NotThisProvider(f"expected 8 kHz μ-law audio, got {media}")
            return Start(stream_id=start.get("streamId") or data.get("streamId") or "", call_id=start.get("callId") or "")

    try:
        return await asyncio.wait_for(read(), timeout_s)
    except asyncio.TimeoutError:
        raise NotThisProvider(f"no start event within {timeout_s:.0f} s") from None


class PlivoLink(PhoneLink):
    """A phone call on Plivo Audio Streams: 8 kHz μ-law, checkpoints, clearAudio."""

    kind = "plivo"

    def __init__(self, websocket, start: Start, pipeline_rate: int, reply_rate: int):
        super().__init__(websocket, start.call_id, start.stream_id, line_rate=PHONE_RATE, encoding=MULAW,
                         pipeline_rate=pipeline_rate, reply_rate=reply_rate)
        self.start = start

    def parse(self, data: Dict[str, Any]):
        event = data.get("event")
        if event == "media":
            return "media", (data.get("media") or {}).get("payload")
        if event == "playedStream":
            return "played", data.get("name")
        if event == "stop":
            return "stop", None
        if event == "dtmf":
            return "dtmf", (data.get("dtmf") or {}).get("digit")
        return "other", None  # clearedAudio, and anything newer

    def media_message(self, payload_b64: str) -> Dict[str, Any]:
        return {"event": "playAudio",
                "media": {"contentType": CONTENT_TYPE, "sampleRate": PHONE_RATE, "payload": payload_b64}}

    def marker_message(self, name: str) -> Dict[str, Any]:
        return {"event": "checkpoint", "streamId": self.stream_id, "name": name}

    def clear_message(self) -> Dict[str, Any]:
        return {"event": "clearAudio", "streamId": self.stream_id}
