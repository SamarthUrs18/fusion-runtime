"""Twilio: answer a phone number with a fusion agent, on the customer's own Twilio account.

    caller ──phone──► Twilio ──POST /v1/phone/twilio──► fusion   "connect the call to my stream"
                      Twilio ◄═══ WebSocket /v1/phone/twilio/stream ═══► fusion   the call's audio

1. Someone calls the number. Twilio asks the number's voice URL what to do: fusion checks the
   request really is Twilio's (X-Twilio-Signature, made with the account's auth token, which
   only Twilio and this server know), and answers with TwiML: connect the call to a stream,
   with a single-use session token as a parameter. Twilio's stream URL can't carry a query
   string or headers, so the token rides in the stream's start message.
2. Twilio opens the WebSocket and sends `connected`, then `start` with the token. From then on
   `media` messages carry 20 ms of the caller's audio (8 kHz μ-law, base64), and fusion sends the
   agent's audio back the same way. Twilio plays what it's sent in order and echoes each `mark`
   once the audio before it has played: that's how fusion knows the caller is still hearing the
   agent, which interruption depends on. `clear` drops everything not yet played. `stop`, or
   fusion closing the socket, ends the call.

The protocol is Twilio's, documented at twilio.com/docs/voice/media-streams; this is our own
implementation of it.
"""
import asyncio
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit
from xml.sax.saxutils import quoteattr

import numpy as np

from fusion_runtime.phone import mulaw
from fusion_runtime.security.keys import Principal
from fusion_runtime.transport import Link, Received

AUTH_TOKEN_ENV = "FUSION_TWILIO_AUTH_TOKEN"  # the account's auth token: Twilio signs every webhook with it
WEBHOOK_PATH = "/v1/phone/twilio"
STREAM_PATH = "/v1/phone/twilio/stream"
PHONE_RATE = 8000
FRAME_BYTES = 160  # 20 ms of 8 kHz μ-law, the size Twilio sends its own audio in
START_TIMEOUT_S = 10.0

# Calls arrive as this, for limits and logs: Twilio authenticates them, not one of our keys
PRINCIPAL = Principal(fingerprint="twilio", name="twilio", via="twilio")


class NotTwilio(Exception):
    """A request or stream that isn't from Twilio, or doesn't follow its protocol."""


# ---- the webhook -------------------------------------------------------------------------------

def signature(auth_token: str, url: str, params: Mapping[str, str]) -> str:
    """X-Twilio-Signature: HMAC-SHA1 of the full URL and the POST parameters sorted by name, base64."""
    payload = url + "".join(f"{name}{params[name]}" for name in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def valid_signature(auth_token: str, url: str, params: Mapping[str, str], given: Optional[str]) -> bool:
    """Whether Twilio made this request. The URL is compared with and without its default port:
    Twilio signs the address as configured, and proxies differ in whether they keep :443."""
    if not given or not auth_token:
        return False
    return any(hmac.compare_digest(signature(auth_token, candidate, params), given)
               for candidate in _url_variants(url))


def _url_variants(url: str):
    parts = urlsplit(url)
    yield url
    default = {"https": 443, "http": 80}.get(parts.scheme)
    host = parts.hostname or ""
    if parts.port == default:
        yield urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    elif parts.port is None and default:
        yield urlunsplit((parts.scheme, f"{host}:{default}", parts.path, parts.query, parts.fragment))


def stream_url(webhook_url: str) -> str:
    """Where Twilio should stream the call: this server's stream endpoint, ws(s) for http(s)."""
    parts = urlsplit(webhook_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, STREAM_PATH, "", ""))


def connect_twiml(url: str, parameters: Mapping[str, str]) -> str:
    """TwiML that connects the call to our stream; it ends when the stream does."""
    params = "".join(f"<Parameter name={quoteattr(k)} value={quoteattr(str(v))}/>" for k, v in parameters.items())
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            f"<Response><Connect><Stream url={quoteattr(url)}>{params}</Stream></Connect></Response>")


def refuse_twiml(message: str) -> str:
    """Said to the caller when fusion can't take the call (at capacity), then hang up."""
    from xml.sax.saxutils import escape

    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Say>{escape(message)}</Say><Hangup/></Response>'


# ---- the stream --------------------------------------------------------------------------------

@dataclass
class Start:
    stream_sid: str
    call_sid: str
    parameters: Dict[str, str] = field(default_factory=dict)


async def read_start(websocket, timeout_s: float = START_TIMEOUT_S) -> Start:
    """Twilio's first messages: `connected`, then `start`. Anything else first isn't Twilio."""
    async def read():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise NotTwilio("the stream closed before it started")
            try:
                data = json.loads(message.get("text") or "")
            except ValueError:
                raise NotTwilio("the stream sent something other than Twilio's JSON messages") from None
            event = data.get("event")
            if event == "connected":
                continue
            if event != "start":
                raise NotTwilio(f"expected Twilio's start message, got {event!r}")
            start = data.get("start") or {}
            media = start.get("mediaFormat") or {}
            if media and (media.get("encoding") != "audio/x-mulaw" or int(media.get("sampleRate", 0)) != PHONE_RATE):
                raise NotTwilio(f"expected 8 kHz μ-law audio, got {media}")
            return Start(stream_sid=data.get("streamSid") or start.get("streamSid") or "",
                         call_sid=start.get("callSid") or "",
                         parameters={str(k): str(v) for k, v in (start.get("customParameters") or {}).items()})

    try:
        return await asyncio.wait_for(read(), timeout_s)
    except asyncio.TimeoutError:
        raise NotTwilio(f"no start message within {timeout_s:.0f} s") from None


class TwilioLink(Link):
    """A phone call on Twilio Media Streams, as a fusion session sees it."""

    kind = "twilio"

    def __init__(self, websocket, start: Start, pipeline_rate: int, reply_rate: int):
        super().__init__(websocket)
        from fusion_runtime.audio.echo_canceller import StreamingResampler

        self.start = start
        self._reply_rate = reply_rate
        self._inbound = StreamingResampler(PHONE_RATE, pipeline_rate)
        self._outbound = StreamingResampler(reply_rate, PHONE_RATE)
        self._unplayed_marks = 0  # marks sent that Twilio hasn't played up to yet
        self._mark_count = 0
        self._leftover = b""  # outbound μ-law short of a whole 20 ms frame

    def describe(self) -> Dict[str, Any]:
        return {"transport": self.kind, "call_sid": self.start.call_sid}

    async def receive(self) -> Received:
        while True:
            message = await self._message()
            text = message.get("text") or ""
            try:
                data = json.loads(text)
            except ValueError:
                return "bad", None, len(text)
            event = data.get("event")
            if event == "media":
                payload = (data.get("media") or {}).get("payload") or ""
                try:
                    phone = mulaw.decode(base64.b64decode(payload))
                except ValueError:
                    return "bad", None, len(text)
                pcm = self._inbound.process(phone.astype(np.float64))
                return "audio", np.clip(np.round(pcm), -32768, 32767).astype(np.int16).tobytes(), len(text)
            if event == "mark":
                self._unplayed_marks = max(0, self._unplayed_marks - 1)
                if self._unplayed_marks == 0:
                    self.controls({"type": "playback", "playing": False})
                continue
            if event == "stop":
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect(1000)  # the caller hung up
            if event == "dtmf" and self.trace is not None:
                self.trace.event("phone.dtmf", stage="audio", digit=(data.get("dtmf") or {}).get("digit"))
            # connected / start again / anything newer: nothing to do

    async def send_audio(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        phone = self._outbound.process(samples)
        data = self._leftover + mulaw.encode(np.clip(np.round(phone), -32768, 32767))
        whole = len(data) - len(data) % FRAME_BYTES
        self._leftover = data[whole:]
        if not whole:
            return
        if self._unplayed_marks == 0:
            self.controls({"type": "playback", "playing": True})
        for i in range(0, whole, FRAME_BYTES):
            await self.websocket.send_json({"event": "media", "streamSid": self.start.stream_sid,
                                            "media": {"payload": base64.b64encode(data[i:i + FRAME_BYTES]).decode()}})
        self._mark_count += 1
        self._unplayed_marks += 1
        await self.websocket.send_json({"event": "mark", "streamSid": self.start.stream_sid,
                                        "mark": {"name": f"fusion-{self._mark_count}"}})

    def event(self, event: Dict[str, Any]) -> None:
        if event.get("type") == "interrupted":
            # The caller talked over the agent: drop what Twilio hasn't played yet
            self._send_soon({"event": "clear", "streamSid": self.start.stream_sid})
            self._unplayed_marks = 0
            self._leftover = b""
            from fusion_runtime.audio.echo_canceller import StreamingResampler

            self._outbound = StreamingResampler(self._reply_rate, PHONE_RATE)  # no tail of the old reply
            self.controls({"type": "playback", "playing": False})
        # transcripts, traces and audio problems have nowhere to go on a phone line: they're in the log
