"""Reading a `.env` file, so settings and tokens don't have to be exported by hand.

    # .env
    HF_TOKEN=hf_...
    FUSION_TURN_WAIT_MS=800

`frun` reads `.env` from the directory it runs in (and, with an agent file, the
directory that file lives in) before anything else looks at the environment.
The real environment always wins, so a value exported in a shell or set by a
deployment isn't overwritten by a stale file.

Parsing is python-dotenv's, the usual library for this (quotes, `export `,
comments, multi-line values). Keep `.env` out of version control: it holds
secrets. `.env.example` lists what can go in it.
"""
import os
from pathlib import Path
from typing import Iterable, List, Optional

ENV_FILENAME = ".env"


def load_env_file(directories: Optional[Iterable[Path]] = None, environ=None) -> List[str]:
    """Load `.env` from each directory given (default: the current one).

    Returns the names of the variables that were set — never their values,
    which are secrets. Variables already in the environment are left alone.
    """
    from dotenv import dotenv_values

    environ = os.environ if environ is None else environ
    loaded: List[str] = []
    seen = set()
    for directory in directories or [Path.cwd()]:
        path = Path(directory).expanduser() / ENV_FILENAME
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        try:
            values = dotenv_values(path)
        except OSError:
            continue
        for key, value in values.items():
            if key not in environ and value is not None:  # the real environment wins
                environ[key] = value
                loaded.append(key)
    return loaded
