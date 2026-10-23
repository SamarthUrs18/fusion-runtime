"""Exotel's half of a phone call: what its Voicebot applet does, for testing fusion without Exotel.

No webhook: the applet opens the stream URL it was configured with, with the URL's credentials as
`Authorization: Basic` (one of fusion's keys), and `sample-rate` in the query string. Then
`connected`, `start`, the caller's audio as `media` (raw 16-bit little-endian PCM, ~100 ms each),
`mark` echoed once audio has been processed, `clear`, and `stop` with a reason on hang-up.

Written from Exotel's documentation (developer.exotel.com/docs/agentstream), not from fusion's code.
"""
import base64
import json
import uuid

import numpy as np
from phone_call import (  # noqa: F401 - Playback is part of this module's API
    PHONE_RATE,
    Call,
    Playback,
)

CHUNK_MS = 100  # Exotel streams the caller roughly every 100 ms


def connected() -> str:
    return json.dumps({"event": "connected"})


def start(stream_sid: str, call_sid: str, sample_rate: int, caller: str, called: str) -> str:
    return json.dumps({
        "event": "start", "sequence_number": "1", "stream_sid": stream_sid,
        "start": {"stream_sid": stream_sid, "call_sid": call_sid, "account_sid": "exotel-test",
                  "from": caller, "to": called, "custom_parameters": {},
                  "media_format": {"encoding": "audio/x-raw", "sample_rate": str(sample_rate), "bit_rate": "16"}},
    })


def media(stream_sid: str, pcm: bytes, sequence: int, chunk: int, timestamp_ms: int) -> str:
    return json.dumps({"event": "media", "sequence_number": str(sequence), "stream_sid": stream_sid,
                       "media": {"chunk": str(chunk), "timestamp": str(timestamp_ms),
                                 "payload": base64.b64encode(pcm).decode()}})


def mark(stream_sid: str, name: str, sequence: int) -> str:
    return json.dumps({"event": "mark", "sequence_number": str(sequence), "stream_sid": stream_sid,
                       "mark": {"name": name}})


def stop(stream_sid: str, call_sid: str, sequence: int, reason: str = "callended") -> str:
    return json.dumps({"event": "stop", "sequence_number": str(sequence), "stream_sid": stream_sid,
                       "stop": {"call_sid": call_sid, "account_sid": "exotel-test", "reason": reason}})


class ExotelCall(Call):
    provider = "exotel"

    def __init__(self, *args, sample_rate: int = PHONE_RATE, **kwargs):
        super().__init__(*args, **kwargs)
        from fusion_runtime.audio.echo_canceller import StreamingResampler

        self.sample_rate = sample_rate
        self._up = StreamingResampler(PHONE_RATE, sample_rate)  # the caller, at the stream's rate
        self._down = StreamingResampler(sample_rate, PHONE_RATE)  # the agent, as the phone plays it
        self._pending = b""
        self.chunk = 0

    async def send_caller_audio(self, samples: np.ndarray) -> None:
        pcm = np.clip(np.round(self._up.process(np.asarray(samples, dtype=np.float64))), -32768, 32767)
        self._pending += pcm.astype("<i2").tobytes()
        chunk_bytes = self.sample_rate * 2 * CHUNK_MS // 1000
        while len(self._pending) >= chunk_bytes:
            self.chunk += 1
            await self.ws.send(media(self.stream_id, self._pending[:chunk_bytes], self.next_sequence(), self.chunk,
                                     self.chunk * CHUNK_MS))
            self._pending = self._pending[chunk_bytes:]

    def read(self, message: dict):
        event = message.get("event")
        if event == "media":
            pcm = base64.b64decode(message["media"]["payload"])
            if len(pcm) < 3200 or len(pcm) > 100_000 or len(pcm) % 320:  # Exotel's rules for what it's sent
                self.on_note(f"fusion sent {len(pcm)} bytes of audio: Exotel wants 3200-100000, a multiple of 320")
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
            phone = np.clip(np.round(self._down.process(samples)), -32768, 32767)
            return "audio", phone.astype(np.int16).tobytes()
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


async def place_call(base_url: str, key: str, playback: Playback, caller: str = "+919800000001",
                     called: str = "+918000000999", sample_rate: int = PHONE_RATE, **call_options):
    """What the Voicebot applet does: open the configured stream URL, credentials as Basic auth."""
    import websockets

    url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    url += f"/v1/phone/exotel/stream?sample-rate={sample_rate}"
    credentials = base64.b64encode(f"fusion:{key}".encode()).decode()
    try:
        ws = await websockets.connect(url, additional_headers={"Authorization": f"Basic {credentials}"})
    except TypeError:  # websockets before 14
        ws = await websockets.connect(url, extra_headers={"Authorization": f"Basic {credentials}"})
    stream_sid, call_sid = "MZ" + uuid.uuid4().hex, "exotel-" + uuid.uuid4().hex
    await ws.send(connected())
    await ws.send(start(stream_sid, call_sid, sample_rate, caller, called))
    return ExotelCall(ws, stream_sid, call_sid, playback, sample_rate=sample_rate, **call_options)
