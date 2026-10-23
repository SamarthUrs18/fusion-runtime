"""What every phone provider's stream has in common, whatever its message format.

A provider streams the caller's audio in small JSON messages, plays what we send back in order,
and tells us when it has played up to a point we marked. The audio is narrowband (8 kHz, μ-law or
16-bit PCM). Each provider (twilio.py, exotel.py, plivo.py) only says how its messages look; the
audio conversion, framing, playback tracking and interruption live here.

Playback tracking is what interruption depends on: the session may only treat the caller's voice as
talking over the agent while the agent is still audible. After each reply chunk we send a marker
(Twilio and Exotel call it a mark, Plivo a checkpoint); while any is unplayed, the agent is playing.
"""
import asyncio
import base64
import contextlib
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Here, not when a call connects: the first import of the resampler (scipy's filter design) took a
# second, and on the event loop that froze audio for every call on the server (seen 24 Oct)
from fusion_runtime.audio.echo_canceller import StreamingResampler
from fusion_runtime.phone import mulaw
from fusion_runtime.transport import Link, Received

MULAW, PCM16 = "mulaw", "pcm16"
TAIL_S = 0.06  # no more of the reply for this long: it has ended, so its last partial frame goes out


class NotThisProvider(Exception):
    """A stream that isn't from the provider it claims to be, or doesn't follow its protocol."""


class PhoneLink(Link):
    phone_line = True
    kind = "phone"

    def __init__(self, websocket, call_id: str, stream_id: str, *, line_rate: int, encoding: str,
                 pipeline_rate: int, reply_rate: int, frame_ms: int = 20):
        super().__init__(websocket)
        self.call_id, self.stream_id = call_id, stream_id
        self.line_rate, self.encoding = line_rate, encoding
        self._reply_rate = reply_rate
        self._inbound = StreamingResampler(line_rate, pipeline_rate)
        self._outbound = StreamingResampler(reply_rate, line_rate)
        bytes_per_sample = 1 if encoding == MULAW else 2
        self.frame_bytes = line_rate * frame_ms // 1000 * bytes_per_sample
        self._unplayed = 0  # markers sent that the provider hasn't played up to yet
        self._markers = 0
        self._leftover = b""  # outbound audio short of a whole frame
        self._silence = mulaw.SILENCE if encoding == MULAW else b"\x00\x00"
        self._tail: Optional[asyncio.Task] = None  # sends the leftover once the reply has stopped coming
        self._sent_unmarked = False  # frames went out after the last marker

    def describe(self) -> Dict[str, Any]:
        return {"transport": self.kind, "call_sid": self.call_id}

    # -- the provider's format: overridden ---------------------------------------------------------

    def parse(self, data: Dict[str, Any]) -> Tuple[str, Any]:
        """One message from the provider, as ("media", bytes) / ("played", name) / ("stop", None) /
        ("dtmf", digit) / ("other", None)."""
        raise NotImplementedError

    def media_message(self, payload_b64: str) -> Dict[str, Any]:
        raise NotImplementedError

    def marker_message(self, name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def clear_message(self) -> Dict[str, Any]:
        raise NotImplementedError

    # -- shared --------------------------------------------------------------------------------------

    async def receive(self) -> Received:
        while True:
            message = await self._message()
            text = message.get("text") or ""
            try:
                data = json.loads(text)
            except ValueError:
                return "bad", None, len(text)
            if not isinstance(data, dict):
                return "bad", None, len(text)
            kind, value = self.parse(data)
            if kind == "media":
                try:
                    line = self._decode(base64.b64decode(value or ""))
                except ValueError:
                    return "bad", None, len(text)
                pcm = self._inbound.process(line)
                return "audio", np.clip(np.round(pcm), -32768, 32767).astype(np.int16).tobytes(), len(text)
            if kind == "played":
                self._unplayed = max(0, self._unplayed - 1)
                if self._unplayed == 0:
                    self.controls({"type": "playback", "playing": False})
            elif kind == "stop":
                from fastapi import WebSocketDisconnect

                raise WebSocketDisconnect(1000)  # the caller hung up
            elif kind == "dtmf" and self.trace is not None:
                self.trace.event("phone.dtmf", stage="audio", digit=value)

    async def send_audio(self, pcm: bytes) -> None:
        """Whole frames now; what's left over waits for the next piece of the reply, and goes out
        padded to a frame once no more comes (TAIL_S). A short frame mid-reply makes some providers
        pause (Exotel wants 3200 bytes at least); dropping the end would cut the last word."""
        if self._tail is not None:
            self._tail.cancel()
            self._tail = None
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        data = self._leftover + self._encode(self._outbound.process(samples))
        whole = len(data) - len(data) % self.frame_bytes
        self._leftover = data[whole:]
        await self._send_frames([data[i:i + self.frame_bytes] for i in range(0, whole, self.frame_bytes)],
                                marker=not self._leftover)
        if self._leftover:
            self._tail = asyncio.get_running_loop().create_task(self._send_tail())

    async def _send_tail(self) -> None:
        await asyncio.sleep(TAIL_S)
        frame, self._leftover = self._leftover, b""
        if frame:
            frame += self._silence * ((self.frame_bytes - len(frame)) // len(self._silence))
            with contextlib.suppress(Exception):  # the caller may have gone
                await self._send_frames([frame], marker=True)

    async def _send_frames(self, frames, marker: bool) -> None:
        if frames and self._unplayed == 0:
            self.controls({"type": "playback", "playing": True})
        for frame in frames:
            await self.websocket.send_json(self.media_message(base64.b64encode(frame).decode()))
        if marker and (frames or self._sent_unmarked):
            self._markers += 1
            self._unplayed += 1
            self._sent_unmarked = False
            await self.websocket.send_json(self.marker_message(f"fusion-{self._markers}"))
        elif frames:
            self._sent_unmarked = True

    def event(self, event: Dict[str, Any]) -> None:
        if event.get("type") == "interrupted":
            # The caller talked over the agent: drop what the provider hasn't played yet
            if self._tail is not None:
                self._tail.cancel()
                self._tail = None
            self._send_soon(self.clear_message())
            self._unplayed = 0
            self._sent_unmarked = False
            self._leftover = b""
            self._outbound = StreamingResampler(self._reply_rate, self.line_rate)  # no tail of the old reply
            self.controls({"type": "playback", "playing": False})
        # transcripts, traces and audio problems have nowhere to go on a phone line: they're in the log

    def _decode(self, data: bytes) -> np.ndarray:
        if self.encoding == MULAW:
            return mulaw.decode(data).astype(np.float64)
        return np.frombuffer(data[: len(data) - len(data) % 2], dtype="<i2").astype(np.float64)

    def _encode(self, samples: np.ndarray) -> bytes:
        clipped = np.clip(np.round(samples), -32768, 32767)
        if self.encoding == MULAW:
            return mulaw.encode(clipped)
        return clipped.astype("<i2").tobytes()


def number(value, default: int = 0) -> int:
    """Providers' numbers arrive as numbers or as numeric strings, depending on the page you read."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
