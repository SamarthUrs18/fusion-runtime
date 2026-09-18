"""Which web pages may open a socket to this server.

A browser tells the server where the page came from:

    Origin: https://shopkart.example

Unlike normal web requests, browsers do not stop a page on one site from opening
a WebSocket to another — the check has to happen here. Without it, any site can
embed the client, point it at your server and spend your GPU on their visitors.

The default is same-origin: the console this runtime serves works, and nothing
else does until you say so. Clients that aren't browsers — `frun talk`, a
backend, curl — send no Origin at all and are unaffected; they hold a key, which
is the real control.
"""
import os
from typing import Iterable, List, Optional
from urllib.parse import urlsplit

ALLOWED_ORIGINS_ENV = "FUSION_ALLOWED_ORIGINS"


def _normalise(origin: str) -> str:
    origin = origin.strip().rstrip("/")
    if not origin:
        return ""
    if "//" not in origin:  # "shopkart.example" means "however it is served"
        return origin.lower()
    parts = urlsplit(origin)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


class OriginRule:
    def __init__(self, allowed: Iterable[str] = ()):
        self.allowed: List[str] = [_normalise(o) for o in allowed if _normalise(o)]
        self.any_origin = "*" in self.allowed

    @property
    def configured(self) -> bool:
        return bool(self.allowed)

    def permits(self, origin: Optional[str], host: Optional[str] = None) -> bool:
        """`host` is what the request was addressed to, which is how same-origin
        is recognised when nothing has been configured."""
        if not origin:
            return True  # not a browser
        if self.any_origin:
            return True
        candidate = _normalise(origin)
        if candidate in self.allowed:
            return True
        if any(a == urlsplit(candidate).netloc for a in self.allowed):  # host written without a scheme
            return True
        if not self.configured and host:
            return urlsplit(candidate).netloc.lower() == host.strip().lower()
        return False

    @classmethod
    def from_environment(cls, environ=None) -> "OriginRule":
        environ = os.environ if environ is None else environ
        return cls((environ.get(ALLOWED_ORIGINS_ENV) or "").split(","))
