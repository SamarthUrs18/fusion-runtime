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
import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

PHONE_RATE = 8000
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

@dataclass
class Playback:
    """What the caller's phone has been sent and not yet played, as Twilio keeps it.

    `queued_s` tells how much audio is ahead of a new mark. Real speakers report that themselves
    (the interactive simulator passes DuplexAudio's); without one, audio plays in real time on a
    clock, which is what a recording or a test needs.
    """
    queued_s: Optional[Callable[[], float]] = None
    _until: float = 0.0  # when the audio sent so far finishes, without a speaker
    marks: List[Tuple[float, str]] = field(default_factory=list)  # (plays at, name)

    def queue(self, seconds: float) -> None:
        self._until = max(self._until, time.monotonic()) + seconds

    def ahead_s(self) -> float:
        return self.queued_s() if self.queued_s is not None else max(0.0, self._until - time.monotonic())

    def add_mark(self, name: str) -> None:
        self.marks.append((time.monotonic() + self.ahead_s(), name))

    def due(self) -> List[str]:
        now = time.monotonic()
        ready = [name for at, name in self.marks if at <= now]
        self.marks = [(at, name) for at, name in self.marks if at > now]
        return ready

    def clear(self) -> List[str]:
        """Drop what hasn't played; the marks waiting behind it come back at once."""
        self._until = time.monotonic()
        pending, self.marks = [name for _, name in self.marks], []
        return pending


class Call:
    """One call to fusion, from Twilio's side of the stream."""

    def __init__(self, websocket, stream_sid: str, call_sid: str, playback: Playback,
                 on_audio: Callable[[bytes], None] = lambda pcm: None, on_note: Callable[[str], None] = print,
                 on_clear: Callable[[], None] = lambda: None):
        self.ws = websocket
        self.stream_sid, self.call_sid = stream_sid, call_sid
        self.playback = playback
        self.on_audio = on_audio  # 8 kHz int16 PCM of the agent, as the phone would play it
        self.on_note = on_note
        self.on_clear = on_clear  # the speaker drops what it hasn't played
        self.sequence = 2  # connected and start came first
        self.chunk = 0
        self.agent_audio_s = 0.0
        self.connected_at = time.monotonic()
        self.first_audio_at: Optional[float] = None  # the greeting's first sound, if the agent greets
        self.clears = 0
        self.closed: Optional[str] = None

    async def send_caller_audio(self, mulaw: bytes) -> None:
        """The caller speaking: 20 ms media messages, as Twilio sends them."""
        for i in range(0, len(mulaw) - FRAME_BYTES + 1, FRAME_BYTES):
            self.sequence += 1
            self.chunk += 1
            await self.ws.send(media(self.stream_sid, mulaw[i:i + FRAME_BYTES], self.sequence, self.chunk,
                                     self.chunk * FRAME_MS))

    async def hang_up(self) -> None:
        self.sequence += 1
        await self.ws.send(stop(self.stream_sid, self.call_sid, self.sequence))
        await self.ws.close()

    async def run(self) -> None:
        """Play and acknowledge what fusion sends until the stream closes."""
        from fusion_runtime.phone import mulaw as codec

        echo = asyncio.create_task(self._echo_marks())
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                event = message.get("event")
                if event == "media":
                    audio = base64.b64decode(message["media"]["payload"])
                    if self.first_audio_at is None:
                        self.first_audio_at = time.monotonic()
                        self.on_note(f"first audio {(self.first_audio_at - self.connected_at) * 1000:.0f} ms "
                                     "after the call connected")
                    seconds = len(audio) / PHONE_RATE
                    self.agent_audio_s += seconds
                    self.playback.queue(seconds)
                    self.on_audio(codec.decode(audio).tobytes())
                elif event == "mark":
                    self.playback.add_mark(message["mark"]["name"])
                elif event == "clear":
                    self.clears += 1
                    self.on_clear()
                    self.on_note("fusion cleared the phone's audio (the caller interrupted)")
                    for name in self.playback.clear():
                        await self._send_mark(name)
                else:
                    self.on_note(f"unexpected message from fusion: {event}")
        except Exception as e:  # the stream closing is how fusion ends a call
            self.closed = f"{type(e).__name__}: {e}"
        finally:
            echo.cancel()
            code = getattr(self.ws, "close_code", None)
            self.closed = self.closed or f"fusion closed the stream (code {code})"

    async def _echo_marks(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            for name in self.playback.due():
                await self._send_mark(name)

    async def _send_mark(self, name: str) -> None:
        self.sequence += 1
        await self.ws.send(mark(self.stream_sid, name, self.sequence))


async def place_call(base_url: str, auth_token: str, playback: Playback, caller: str = "+15550100001",
                     called: str = "+15550100999", **call_options):
    """Ring fusion: the signed webhook, then the stream it asks for. Returns the open Call, or raises
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
    return Call(ws, stream_sid, call_sid, playback, **call_options)
