"""Interruption ("barge-in") state shared between generation and the watcher."""
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import time


@dataclass
class BargeInState:
    """Coordinates interruption ("barge-in") between the reply generation
    loop and an independent watcher task that keeps scanning incoming audio
    for real user speech while the bot is talking.

    "Talking" has two parts. `generating` covers the server producing a
    reply; `playing` is reported by the client and covers audio still coming
    out of its speaker — which usually outlasts generation by seconds, since
    text is produced faster than it's spoken. Watching generation alone left
    the end of every reply impossible to interrupt.

    Once an interruption fires, `speaking` stays False until the next reply
    starts, so a single interruption can't fire twice. The watcher still
    requires *sustained* speech, since a client without echo cancellation
    can't promise echo-free audio — see `_barge_in_watcher`.

    `speaking_since` (time.monotonic) is when the bot became audible. The
    watcher can lag behind the audio (model loading, a slow moment, audio
    sent faster than real time), so it compares each frame's *arrival* time
    against this, rather than asking whether the bot is speaking at the
    moment the frame gets processed. Otherwise the user's own words from
    just before the reply started count as talking over the bot.
    """
    generating: bool = False
    playing: bool = False
    interrupted: asyncio.Event = field(default_factory=asyncio.Event)
    speaking_since: Optional[float] = None
    _fired: bool = False

    @property
    def speaking(self) -> bool:
        return (self.generating or self.playing) and not self._fired

    def mark_speaking(self):
        self.generating = True
        self._fired = False
        self.speaking_since = time.monotonic()
        self.interrupted.clear()

    def mark_idle(self):
        self.generating = False
        self.interrupted.clear()

    def set_playing(self, playing: bool):
        if playing and not self.generating and not self.playing and self.speaking_since is None:
            self.speaking_since = time.monotonic()
        self.playing = playing
        if not playing and not self.generating:
            self.speaking_since = None

    def fire(self):
        """Record an interruption: cancels in-flight generation and stops
        watching until the next reply starts."""
        self._fired = True
        self.interrupted.set()
