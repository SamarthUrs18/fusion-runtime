"""What a phone provider does with a call's audio, whichever provider it is.

Each provider's side (twilio_side.py, exotel_side.py, plivo_side.py) says how its messages look and
how a call is placed; this plays what fusion sends, in order, acknowledges each marker once the
audio before it has played, and drops what hasn't played when fusion clears it.

The caller's audio goes in as 8 kHz int16 samples, the phone network's rate; each provider encodes
it its own way (μ-law, or 16-bit PCM at the stream's rate). The agent's audio comes back to
`on_audio` as 8 kHz int16 PCM, as the phone would play it.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

PHONE_RATE = 8000
FRAME_SAMPLES = 160  # 20 ms at 8 kHz


@dataclass
class Playback:
    """What the caller's phone has been sent and not yet played, as the provider keeps it.

    `queued_s` tells how much audio is ahead of a new marker. Real speakers report that themselves
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
        """Drop what hasn't played; the markers waiting behind it are returned."""
        self._until = time.monotonic()
        pending, self.marks = [name for _, name in self.marks], []
        return pending


class Call:
    """One call to fusion, from the provider's side of the stream. Subclasses speak a provider's format."""

    provider = "phone"

    def __init__(self, websocket, stream_id: str, call_id: str, playback: Playback,
                 on_audio: Callable[[bytes], None] = lambda pcm: None, on_note: Callable[[str], None] = print,
                 on_clear: Callable[[], None] = lambda: None):
        self.ws = websocket
        self.stream_id, self.call_id = stream_id, call_id
        self.playback = playback
        self.on_audio = on_audio
        self.on_note = on_note
        self.on_clear = on_clear  # the speaker drops what it hasn't played
        self.sequence = 2  # connected and start came first
        self.agent_audio_s = 0.0
        self.connected_at = time.monotonic()
        self.first_audio_at: Optional[float] = None  # the greeting's first sound, if the agent greets
        self.clears = 0
        self.closed: Optional[str] = None

    # -- the provider's format: overridden ---------------------------------------------------------

    async def send_caller_audio(self, samples: np.ndarray) -> None:
        """8 kHz int16 samples of the caller, sent the way the provider sends them."""
        raise NotImplementedError

    def read(self, message: dict) -> Tuple[str, object]:
        """("audio", 8 kHz int16 bytes) / ("marker", name) / ("clear", None) / ("other", event name)."""
        raise NotImplementedError

    async def acknowledge(self, name: str) -> None:
        """Tell fusion the audio up to this marker has played."""
        raise NotImplementedError

    async def cleared(self, pending: List[str]) -> None:
        """After a clear: what the provider says back."""
        for name in pending:
            await self.acknowledge(name)

    async def hang_up(self) -> None:
        await self.ws.close()

    # -- shared --------------------------------------------------------------------------------------

    async def run(self) -> None:
        """Play and acknowledge what fusion sends until the stream closes."""
        echo = asyncio.create_task(self._acknowledge_played())
        try:
            async for raw in self.ws:
                kind, value = self.read(json.loads(raw))
                if kind == "audio":
                    if self.first_audio_at is None:
                        self.first_audio_at = time.monotonic()
                        self.on_note(f"first audio {(self.first_audio_at - self.connected_at) * 1000:.0f} ms "
                                     "after the call connected")
                    seconds = len(value) / 2 / PHONE_RATE
                    self.agent_audio_s += seconds
                    self.playback.queue(seconds)
                    self.on_audio(value)
                elif kind == "marker":
                    self.playback.add_mark(value)
                elif kind == "clear":
                    self.clears += 1
                    self.on_clear()
                    self.on_note("fusion cleared the phone's audio (the caller interrupted)")
                    await self.cleared(self.playback.clear())
                else:
                    self.on_note(f"unexpected message from fusion: {value}")
        except Exception as e:  # the stream closing is how fusion ends a call
            self.closed = f"{type(e).__name__}: {e}"
        finally:
            echo.cancel()
            code = getattr(self.ws, "close_code", None)
            self.closed = self.closed or f"fusion closed the stream (code {code})"

    async def _acknowledge_played(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            for name in self.playback.due():
                await self.acknowledge(name)

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence
