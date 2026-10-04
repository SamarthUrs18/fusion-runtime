"""The welcome screen `frun up` shows in a terminal.

Only for a person looking at a terminal: never in JSON logs, a container's log, a
file or a pipe, where it would be noise a log collector has to skip. Narrow windows
get the one-line form, so the letters never wrap into a mess. NO_COLOR is honoured.
"""
import os
import sys
from typing import List, Mapping, Optional, Sequence, TextIO, Tuple

# Brand colours (the website's): orange shading to amber down the word, warm grey for the wave
SHADES = (
    "\033[38;2;217;97;47m", "\033[38;2;222;108;50m", "\033[38;2;226;120;54m",
    "\033[38;2;230;131;58m", "\033[38;2;234;142;62m",
)
ORANGE, AMBER = SHADES[0], "\033[38;2;232;140;60m"
GREY = "\033[38;2;140;130;120m"
DIM = "\033[2m"
RESET = "\033[0m"

# "fusion" in lowercase slanted strokes, like the website's wordmark. Thin lines on purpose:
# solid orange blocks read as other tools' logos.
WORD = (
    "     ____           _          ",
    "    / __/_  _______(_)___  ____",
    "   / /_/ / / / ___/ / __ \\/ __ \\",
    "  / __/ /_/ (__  ) / /_/ / / / /",
    " /_/  \\__,_/____/_/\\____/_/ /_/",
)
# A voice, as fine dots; terminals that can't draw them get thin bars, then nothing
WAVES = ("⣀⣠⣴⣾⣷⣦⣄⣀⣠⣴⣦⣄", "ılıllıılıllı", "")
TAGLINE = "self-hosted voice agents on your own GPU"
WIDE_ENOUGH = max(len(row) for row in WORD) + 4  # the word plus its margin


def wave_for(encoding: Optional[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """The finest wave this terminal can draw. The old Windows console has no font for the dots."""
    env = os.environ if environ is None else environ
    legacy_windows = os.name == "nt" and not env.get("WT_SESSION")  # Windows Terminal sets WT_SESSION
    for wave in WAVES[1:] if legacy_windows else WAVES:
        try:
            wave.encode(encoding or "ascii")
            return wave
        except (UnicodeEncodeError, LookupError):
            continue
    return ""


def should_show(stream: TextIO, json_logs: bool, environ: Optional[Mapping[str, str]] = None) -> bool:
    """A person is reading this: a real terminal, readable logs, and a terminal that can draw."""
    env = os.environ if environ is None else environ
    if json_logs or env.get("TERM") == "dumb":
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def render(version: str, rows: Sequence[Tuple[str, str]], notes: Sequence[str] = (), width: int = 80,
           color: bool = True, wave: str = WAVES[0]) -> str:
    """The welcome screen as text: the word (or one line when narrow), then rows of label and value."""
    def paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if color else text

    lines: List[str] = [""]
    if width >= WIDE_ENOUGH:
        for shade, row in zip(SHADES, WORD, strict=True):
            lines.append("  " + paint(shade, row))
        lead = f"{paint(GREY, wave)}  " if wave else ""
        lines.append(f"  {lead}{paint(AMBER, ' '.join('runtime'))}   {paint(DIM, version)}")
    else:
        lines.append(f"  {paint(ORANGE, 'fusion')}{paint(AMBER, '-runtime')} {paint(DIM, version)}")
    lines.append("")
    lines.append("  " + paint(DIM, TAGLINE))
    lines.append("")
    label_width = max((len(label) for label, _ in rows), default=0) + 2
    for label, value in rows:
        lines.append(f"  {paint(ORANGE, label.ljust(label_width))}{value}")
    for note in notes:
        lines.append(f"  {' ' * label_width}{paint(DIM, note)}")
    lines.append("")
    return "\n".join(lines)


def show(version: str, rows: Sequence[Tuple[str, str]], notes: Sequence[str] = (),
         stream: Optional[TextIO] = None, environ: Optional[Mapping[str, str]] = None) -> None:
    import shutil

    out = stream or sys.stdout
    env = os.environ if environ is None else environ
    color = not env.get("NO_COLOR")  # https://no-color.org
    out.write(render(version, rows, notes, width=shutil.get_terminal_size((80, 24)).columns, color=color,
                     wave=wave_for(getattr(out, "encoding", None), env)) + "\n")
    out.flush()
