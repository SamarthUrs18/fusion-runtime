"""What one caller may take.

Authentication says who may use the server; it doesn't stop a legitimate key
from opening fifty sockets, streaming audio forever, or connecting in a loop.
This runtime serves few conversations at once, so one client exhausting it is an
outage for everyone else — on rented hardware, an expensive one.

Everything here is a ceiling with a plain default, not a policy engine. Over a
limit is a clear close and a telemetry event, never a silent hang.
"""
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class Limits:
    max_sessions: int = 4  # conversations at once, whole server
    max_sessions_per_key: int = 0  # 0 = derived: see per_key()
    max_message_bytes: int = MEGABYTE  # audio frames are ~1 KB; anything this big is an attack
    max_turn_audio_s: float = 60.0  # one person talking without ever pausing
    max_session_s: float = 900.0  # wall clock for one conversation
    idle_timeout_s: float = 60.0  # no audio and no messages: reclaim the slot
    connections_per_minute: int = 30  # per address, before authentication
    tokens_per_minute: int = 600  # session tokens one key may mint; generous, but not unlimited

    @classmethod
    def from_environment(cls, environ=None) -> "Limits":
        environ = os.environ if environ is None else environ

        def number(name: str, fallback):
            raw = environ.get(name)
            if raw is None or raw == "":
                return fallback
            try:
                return type(fallback)(raw)
            except ValueError:
                return fallback

        return cls(
            max_sessions=number("FUSION_MAX_SESSIONS", cls.max_sessions),
            max_sessions_per_key=number("FUSION_MAX_SESSIONS_PER_KEY", cls.max_sessions_per_key),
            max_message_bytes=number("FUSION_MAX_MESSAGE_BYTES", cls.max_message_bytes),
            max_turn_audio_s=number("FUSION_MAX_TURN_AUDIO_S", cls.max_turn_audio_s),
            max_session_s=number("FUSION_MAX_SESSION_S", cls.max_session_s),
            idle_timeout_s=number("FUSION_IDLE_TIMEOUT_S", cls.idle_timeout_s),
            connections_per_minute=number("FUSION_CONNECTIONS_PER_MINUTE", cls.connections_per_minute),
            tokens_per_minute=number("FUSION_TOKENS_PER_MINUTE", cls.tokens_per_minute),
        )

    def per_key(self, keys: int = 1) -> int:
        """How many conversations one key may run at once.

        Set it explicitly and that wins. Otherwise: one key gets the whole
        server, because a tighter default would silently cap a deployment that
        has nothing to share with. As soon as keys *are* shared, one of them
        keeping a slot free for the others matters more than its own ceiling.
        """
        if self.max_sessions_per_key:
            return self.max_sessions_per_key
        return self.max_sessions if keys <= 1 else max(1, self.max_sessions - 1)


class OverLimit(Exception):
    """A ceiling was reached. `reason` is for telemetry, `message` for the client."""

    def __init__(self, reason: str, message: str, close_code: int = 1013):
        super().__init__(message)
        self.reason = reason
        self.close_code = close_code


class ConnectionRate:
    """A sliding window per address, checked before authentication — otherwise
    guessing keys costs an attacker nothing but a loop."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute = per_minute
        self._clock = clock
        self._seen: Dict[str, Deque[float]] = {}

    def check(self, address: Optional[str]) -> None:
        if self.per_minute <= 0 or address is None:
            return
        now = self._clock()
        window = self._seen.setdefault(address, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            raise OverLimit("connection_rate", "too many connections from this address; try again shortly")
        window.append(now)
        if len(self._seen) > 10_000:  # a scan from many addresses shouldn't grow this without bound
            self._forget_quiet(now)

    def _forget_quiet(self, now: float) -> None:
        for address in [a for a, w in self._seen.items() if not w or now - w[-1] > 60]:
            del self._seen[address]


class SessionSlots:
    """How many conversations are running, in total and per key."""

    def __init__(self, limits: Limits, keys: int = 1):
        self.limits = limits
        self.keys = keys
        self._total = 0
        self._per_key: Dict[str, int] = {}

    def take(self, key: str) -> None:
        if self._total >= self.limits.max_sessions:
            raise OverLimit("server_full",
                            "this server is at capacity; try again in a moment")
        if self._per_key.get(key, 0) >= self.limits.per_key(self.keys):
            raise OverLimit("key_at_limit",
                            "this key already has as many conversations as it may run at once")
        self._total += 1
        self._per_key[key] = self._per_key.get(key, 0) + 1

    def give_back(self, key: str) -> None:
        self._total = max(0, self._total - 1)
        remaining = self._per_key.get(key, 1) - 1
        if remaining > 0:
            self._per_key[key] = remaining
        else:
            self._per_key.pop(key, None)

    @property
    def in_use(self) -> int:
        return self._total


class AudioBudget:
    """Bytes and seconds one socket may use.

    A turn that never ends, a session that never ends, and a socket that opened
    and went quiet are three different ways to hold a slot for free.
    """

    def __init__(self, limits: Limits, sample_rate: int, clock=time.monotonic):
        self.limits = limits
        self.bytes_per_second = max(1, sample_rate * 2)  # 16-bit mono
        self._clock = clock
        self.started = clock()
        self.last_activity = self.started
        self.turn_bytes = 0

    def message(self, size: int) -> None:
        if size > self.limits.max_message_bytes:
            raise OverLimit("message_too_big",
                            f"a message of {size} bytes is over the {self.limits.max_message_bytes} byte limit",
                            close_code=1009)
        self.last_activity = self._clock()

    def audio(self, size: int) -> None:
        self.message(size)
        self.turn_bytes += size
        if self.turn_bytes / self.bytes_per_second > self.limits.max_turn_audio_s:
            raise OverLimit("turn_too_long",
                            f"more than {self.limits.max_turn_audio_s:.0f} seconds of speech without a pause",
                            close_code=1008)

    def turn_ended(self) -> None:
        self.turn_bytes = 0

    def expired(self) -> Optional[OverLimit]:
        now = self._clock()
        if now - self.started > self.limits.max_session_s:
            return OverLimit("session_too_long",
                             f"conversations are limited to {self.limits.max_session_s / 60:.0f} minutes",
                             close_code=1000)
        if now - self.last_activity > self.limits.idle_timeout_s:
            return OverLimit("idle",
                             f"nothing received for {self.limits.idle_timeout_s:.0f} seconds",
                             close_code=1000)
        return None
