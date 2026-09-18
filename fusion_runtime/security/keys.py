"""The keys a server accepts, and who presented one.

A key is a secret string the operator generates (`frun key new`) and puts in the
environment — never in an agent file, never in the repo:

    FUSION_ACCEPTED_KEYS=web:frun_kR7m…,mobile:frun_9xQ2…
    FUSION_ACCEPTED_KEYS_FILE=/etc/fusion/keys      # one per line, reloadable

The name in front is a label for the operator, not a role: it carries no
permissions, and exists so logs can say *which* key is busy or failing without
ever printing the key itself. Where no name is given, the fingerprint — the
first eight characters of the key's SHA-256 — identifies it, and `frun key new`
prints the same value, so an operator can match a line in the logs to the entry
in their password manager without either of us handling the secret.
"""
import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

KEY_PREFIX = "frun_"
# Short enough to brute force is worse than no key at all, because it reads as
# protection. This is the one moment we can refuse "FUSION_ACCEPTED_KEYS=secret123".
MIN_KEY_LENGTH = 16


class ConfigurationError(ValueError):
    """The keys themselves are wrong — the server should refuse to start."""


@dataclass(frozen=True)
class Principal:
    """Who is making a request: a named key, or the token it minted."""

    fingerprint: str
    name: Optional[str] = None
    via: str = "key"  # key | token | loopback

    @property
    def label(self) -> str:
        return self.name or self.fingerprint


def fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


def _parse_entry(raw: str, source: str) -> Optional[Tuple[Optional[str], str]]:
    entry = raw.strip()
    if not entry or entry.startswith("#"):
        return None
    name: Optional[str] = None
    if ":" in entry:
        name, _, entry = entry.partition(":")
        name, entry = name.strip(), entry.strip()
    if len(entry) < MIN_KEY_LENGTH:
        raise ConfigurationError(
            f"a key in {source} is only {len(entry)} characters. Keys must be at least "
            f"{MIN_KEY_LENGTH}, or they aren't protecting anything. Generate one: frun key new"
        )
    return (name or None, entry)


class KeySet:
    """The keys this server accepts, and nothing more.

    Verification walks every key rather than returning on the first match, and
    compares with `hmac.compare_digest`, so a wrong key can't be narrowed down
    by timing.
    """

    def __init__(self, entries: Iterable[Tuple[Optional[str], str]] = ()):
        self._entries: List[Tuple[Optional[str], str]] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return bool(self._entries)

    def verify(self, presented: Optional[str]) -> Optional[Principal]:
        if not presented:
            return None
        found: Optional[Principal] = None
        for name, secret in self._entries:
            if hmac.compare_digest(presented, secret) and found is None:
                found = Principal(fingerprint=fingerprint(secret), name=name, via="key")
        return found

    def secrets(self) -> List[str]:
        """The keys themselves. Only for a client presenting one to a local server."""
        return [secret for _, secret in self._entries]

    def describe(self) -> List[Dict[str, str]]:
        """Names and fingerprints, for `frun keys list`. Never the keys."""
        return [{"name": name or "-", "fingerprint": fingerprint(secret)} for name, secret in self._entries]

    @classmethod
    def from_environment(cls, environ=None) -> "KeySet":
        from fusion_runtime.config import ACCEPTED_KEYS_ENV, ACCEPTED_KEYS_FILE_ENV

        environ = os.environ if environ is None else environ
        entries: List[Tuple[Optional[str], str]] = []
        seen = set()
        raw = environ.get(ACCEPTED_KEYS_ENV, "")
        for part in raw.split(","):
            entry = _parse_entry(part, ACCEPTED_KEYS_ENV)
            if entry and entry[1] not in seen:
                seen.add(entry[1])
                entries.append(entry)
        path = environ.get(ACCEPTED_KEYS_FILE_ENV)
        if path:
            entries.extend(e for e in cls._read_file(Path(path)) if e[1] not in seen)
        return cls(entries)

    @staticmethod
    def _read_file(path: Path) -> List[Tuple[Optional[str], str]]:
        """One key per line. This is how Kubernetes and Docker hand a secret to a
        process, and unlike an environment variable it can be re-read without a
        restart — so a leaked key doesn't cost a reload of the models."""
        try:
            lines = path.expanduser().read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise ConfigurationError(f"can't read the keys file {path}: {e}") from e
        found = [_parse_entry(line, str(path)) for line in lines]
        return [entry for entry in found if entry is not None]


def new_key(name: Optional[str] = None) -> str:
    """A fresh key: 256 bits from the OS, with a prefix that makes it recognisable
    in a leaked file and to secret scanners."""
    import secrets

    key = KEY_PREFIX + secrets.token_urlsafe(32)
    return f"{name}:{key}" if name else key


def client_key(url: str = "", environ=None) -> Optional[str]:
    """The key a client should present.

    FUSION_API_KEY is the answer: it is the credential this machine sends. When
    it isn't set and the target is this same machine, the first key the local
    server accepts is used instead — on a development box the two are the same
    string, and typing it twice teaches nothing.

    The loopback condition is the whole safety of that shortcut. Falling back for
    a remote URL would send the keys of *your* server to somebody else's.
    """
    from urllib.parse import urlsplit

    from fusion_runtime.config import API_KEY_ENV

    environ = os.environ if environ is None else environ
    presented = (environ.get(API_KEY_ENV) or "").strip()
    if presented:
        return presented
    host = (urlsplit(url).hostname or "").lower() if url else ""
    if host not in ("127.0.0.1", "::1", "localhost", ""):
        return None
    local = KeySet.from_environment(environ)
    return local.secrets()[0] if len(local) else None
