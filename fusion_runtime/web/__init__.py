"""What a browser gets from the runtime: a console page, and the script to embed.

`frun up` serves two things from this folder:

    GET /                     the console — open it and talk to your agent
    GET /fusion-runtime.js    the client, for your own pages

They are the same client. The console is that script driving a page we ship, so
the thing we demo with is the thing a site embeds, and it can't drift.

The files are read from disk on every request (a few kB), so editing one and
reloading the browser is enough while working on them.
"""
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
CONSOLE_FILE = WEB_DIR / "index.html"
CLIENT_FILE = WEB_DIR / "fusion-runtime.js"
CLIENT_ROUTE = "/fusion-runtime.js"


class WebAssetMissing(RuntimeError):
    """The browser files aren't next to the code — a packaging problem, not a user one."""


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise WebAssetMissing(
            f"{path.name} is missing from the installed package ({path}). "
            "Reinstall fusion-runtime, or run from a checkout."
        ) from e


def console_html() -> str:
    return read(CONSOLE_FILE)


def client_js() -> str:
    return read(CLIENT_FILE)
