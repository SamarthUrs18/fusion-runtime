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

from fusion_runtime.phone.link import MULAW, NotThisProvider, PhoneLink
from fusion_runtime.security.keys import Principal

AUTH_TOKEN_ENV = "FUSION_TWILIO_AUTH_TOKEN"  # the account's auth token: Twilio signs every webhook with it
WEBHOOK_PATH = "/v1/phone/twilio"
STREAM_PATH = "/v1/phone/twilio/stream"
PHONE_RATE = 8000
START_TIMEOUT_S = 10.0

# Calls arrive as this, for limits and logs: Twilio authenticates them, not one of our keys
PRINCIPAL = Principal(fingerprint="twilio", name="twilio", via="twilio")


NotTwilio = NotThisProvider  # a request or stream that isn't from Twilio, or doesn't follow its protocol


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


class TwilioLink(PhoneLink):
    """A phone call on Twilio Media Streams: 8 kHz μ-law, marks, clear."""

    kind = "twilio"

    def __init__(self, websocket, start: Start, pipeline_rate: int, reply_rate: int):
        super().__init__(websocket, start.call_sid, start.stream_sid, line_rate=PHONE_RATE, encoding=MULAW,
                         pipeline_rate=pipeline_rate, reply_rate=reply_rate)
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
        return {"event": "media", "streamSid": self.stream_id, "media": {"payload": payload_b64}}

    def marker_message(self, name: str) -> Dict[str, Any]:
        return {"event": "mark", "streamSid": self.stream_id, "mark": {"name": name}}

    def clear_message(self) -> Dict[str, Any]:
        return {"event": "clear", "streamSid": self.stream_id}
