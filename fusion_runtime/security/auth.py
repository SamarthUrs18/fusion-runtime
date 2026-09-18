"""Deciding whether a request may use this server.

Two paths in, because two kinds of client exist:

    a browser page      → a session token, in the query string
    anything else       → the key itself, in an Authorization header

and one rule when no keys are configured at all: **the server answers only on
localhost**. Development shouldn't need ceremony, but a runtime reachable from
elsewhere with no authentication is someone else's GPU, paid for by you. There
is deliberately no flag to switch that off: a security control with a documented
bypass is a suggestion, and inside a container the process usually has to bind
0.0.0.0, which is exactly where the bypass would end up permanently on.
"""
import ipaddress
import logging
import re
from typing import Optional

from fusion_runtime.security.keys import KeySet, Principal
from fusion_runtime.security.tokens import TokenStore

BEARER = "bearer "
LOOPBACK = Principal(fingerprint="local", name="localhost", via="loopback")


class Unauthorized(Exception):
    """The caller may not use this server. Deliberately vague about why."""

    code = "auth_failed"

    def __init__(self, message: str, fix: Optional[str] = None):
        super().__init__(message)
        self.fix = fix


def bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.lower().startswith(BEARER):
        return None
    return authorization[len(BEARER):].strip() or None


class ProxyTrust:
    """Which peers may speak for someone else.

    `X-Forwarded-For` and `X-Forwarded-Proto` are headers, so anyone who can
    reach the port can write them. They are only meaningful when the connection
    came from a proxy that overwrites whatever the client claimed — so the
    setting names the proxies rather than saying "trust the headers":

        FUSION_TRUSTED_PROXY=10.0.0.0/8,192.168.1.5     the proxies that are yours
        FUSION_TRUSTED_PROXY=1                          any peer (weaker: fine when
                                                        nothing else can reach the port)
    """

    def __init__(self, value: Optional[str] = None):
        self.any_peer = False
        self._networks = []
        for part in (value or "").split(","):
            token = part.strip()
            if not token or token.lower() in ("0", "false", "no", "off"):
                continue
            if token.lower() in ("1", "true", "yes", "on", "*"):
                self.any_peer = True
                continue
            try:
                self._networks.append(ipaddress.ip_network(token, strict=False))
            except ValueError:
                continue  # a typo must not silently become "trust everyone"

    @property
    def enabled(self) -> bool:
        return self.any_peer or bool(self._networks)

    def believes(self, peer: Optional[str]) -> bool:
        if not self.enabled:
            return False
        if self.any_peer:
            return True
        try:
            address = ipaddress.ip_address((peer or "").strip("[]"))
        except ValueError:
            return False
        return any(address in network for network in self._networks)

    @classmethod
    def from_environment(cls, environ=None) -> "ProxyTrust":
        import os

        from fusion_runtime.config import TRUSTED_PROXY_ENV

        environ = os.environ if environ is None else environ
        return cls(environ.get(TRUSTED_PROXY_ENV))


def is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False  # a name, or a test client: treat as remote, never as trusted


class Authenticator:
    def __init__(self, keys: KeySet, tokens: TokenStore):
        self.keys = keys
        self.tokens = tokens

    @property
    def enabled(self) -> bool:
        return self.keys.enabled

    def describe(self) -> str:
        if not self.enabled:
            return "off (this server answers on localhost only)"
        return f"on ({len(self.keys)} key{'s' if len(self.keys) != 1 else ''})"

    def authenticate(self, *, client_host: Optional[str], authorization: Optional[str] = None,
                     token: Optional[str] = None, allow_token: bool = True) -> Principal:
        """Who is this? Raises Unauthorized if the answer is "nobody we accept"."""
        if not self.enabled:
            if is_loopback(client_host):
                return LOOPBACK
            raise Unauthorized(
                "this server has no keys configured, so it answers on localhost only",
                fix="Generate one with `frun key new`, then set FUSION_ACCEPTED_KEYS before "
                    "starting the server.",
            )
        presented = bearer(authorization)
        principal = self.keys.verify(presented)
        if principal is None and allow_token:
            principal = self.tokens.redeem(token)
        if principal is None:
            # One message for a wrong key, an expired token and a reused one: an
            # attacker learns nothing, and an operator has only one thing to check.
            raise Unauthorized(
                "the key or session token wasn't accepted",
                fix="Send a valid key as `Authorization: Bearer <key>`, or a session token "
                    "from POST /v1/sessions. Tokens work once and expire in about a minute.",
            )
        return principal

    def secure_enough_to_mint(self, *, scheme: str, client_host: Optional[str],
                              forwarded_proto: Optional[str] = None, trust_proxy: bool = False) -> bool:
        """A token in a URL over plain http is readable by every hop in between,
        so we hand one out only where the connection is encrypted — or where it
        never leaves the machine."""
        if scheme in ("https", "wss") or is_loopback(client_host):
            return True
        return bool(trust_proxy and forwarded_proto and forwarded_proto.split(",")[0].strip() == "https")


_QUERY_SECRET = re.compile(r"((?:token|key|api_key)=)[^&\s\"']+", re.IGNORECASE)


class RedactQueryStrings(logging.Filter):
    """Keeps session tokens out of the access log.

    A browser can only carry a token in the query string, and uvicorn writes the
    full path of every request — so without this, each connection quietly logs a
    working credential.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                _QUERY_SECRET.sub(r"\1[redacted]", a) if isinstance(a, str) else a for a in record.args
            )
        if isinstance(record.msg, str):
            record.msg = _QUERY_SECRET.sub(r"\1[redacted]", record.msg)
        return True


def scrub_access_logs() -> None:
    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(RedactQueryStrings())
