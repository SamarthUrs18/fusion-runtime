"""Short-lived session tokens, for clients that can't hold a key.

A browser can do neither of the things a key needs: it can't keep a secret (the
page is public) and it can't set a header on a WebSocket handshake. So the page's
own backend — which does hold the key — asks for a token and hands that over:

    POST /v1/sessions   Authorization: Bearer <key>
      → {"token": "...", "expires_in": 60, "ws_url": "wss://…?token=..."}

The token is random, lives about a minute, and is destroyed the moment it is
used. That matters because a token ends up where a key never should: the address
bar, browser history, a screenshot, a proxy log. Single use plus a short life
means every one of those is already worthless by the time anyone reads it. A
reusable token would just be an API key with extra steps.

Tokens are held in memory, which is right for one process and wrong for several:
when the runtime is replicated behind a router, this becomes a shared store or a
signed token. Clients see no difference either way — they receive a string.
"""
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional

from fusion_runtime.security.keys import Principal

DEFAULT_TTL_S = 60
TOKEN_BYTES = 24


@dataclass(frozen=True)
class Minted:
    token: str
    expires_in: int


@dataclass
class _Entry:
    principal: Principal
    expires_at: float


class TokenStore:
    def __init__(self, ttl_s: int = DEFAULT_TTL_S, clock=time.monotonic):
        self.ttl_s = max(1, int(ttl_s))
        self._clock = clock
        self._tokens: Dict[str, _Entry] = {}

    def mint(self, principal: Principal) -> Minted:
        self._purge()
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._tokens[token] = _Entry(principal=principal, expires_at=self._clock() + self.ttl_s)
        return Minted(token=token, expires_in=self.ttl_s)

    def redeem(self, token: Optional[str]) -> Optional[Principal]:
        """Spend a token. Returns who minted it, or None if it is unknown, expired
        or already used — the caller shouldn't be told which."""
        if not token:
            return None
        entry = self._tokens.pop(token, None)  # single use: gone whether or not it was valid
        if entry is None or entry.expires_at <= self._clock():
            return None
        return Principal(fingerprint=entry.principal.fingerprint, name=entry.principal.name, via="token")

    def drop_for_key(self, fingerprint: str) -> int:
        """Forget every token a key minted. Called when that key is removed: a
        token inherits its key, so a revoked key must not keep working for
        another minute through tokens it already issued."""
        doomed = [t for t, e in self._tokens.items() if e.principal.fingerprint == fingerprint]
        for token in doomed:
            del self._tokens[token]
        return len(doomed)

    def _purge(self) -> None:
        now = self._clock()
        for token in [t for t, e in self._tokens.items() if e.expires_at <= now]:
            del self._tokens[token]

    def __len__(self) -> int:
        self._purge()
        return len(self._tokens)
