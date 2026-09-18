"""Log outputs: a readable console for people, JSON lines for log collectors."""
import json
import threading
from datetime import datetime
from typing import Any, TextIO

from fusion_runtime.telemetry.events import Event
from fusion_runtime.telemetry.hub import Sink

# attrs shown first on a console line, in this order; the rest follow alphabetically
_LEADING_ATTRS = ("runtime", "model", "reason", "outcome", "ttfa_ms", "response_ms")


def _human_time(ms: float) -> str:
    seconds = round(ms / 1000)
    return f"{seconds // 60}m {seconds % 60}s" if seconds >= 60 else f"{seconds}s"


def _download_line(event: Event) -> str:
    """Downloads are the one thing that keeps someone waiting, so they read as plain sentences."""
    kind = {"stt": "speech-to-text", "llm": "language", "tts": "voice"}.get(event.stage or "", "")
    what = f"{kind} model {event.attrs.get('model', '')}".strip()
    size = event.attrs.get("size")
    if event.name == "model.downloading":
        return (f"⬇  Downloading the {what}{f' ({size})' if size else ''}. "
                "This happens the first time only. Please wait.")
    took = f" in {_human_time(event.duration_ms)}" if event.duration_ms else ""
    return f"✓  Downloaded the {what}{took}. Next time it starts straight away."


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}"
    if isinstance(value, str) and (" " in value or not value):
        return json.dumps(value)
    return str(value)


class ConsoleSink(Sink):
    def __init__(self, stream: TextIO, min_level: str = "info"):
        self.stream = stream
        self.min_level = min_level
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        with self._lock:
            self.stream.write(self.format(event) + "\n")
            self.stream.flush()

    @staticmethod
    def format(event: Event) -> str:
        clock = datetime.fromtimestamp(event.ts).strftime("%H:%M:%S.%f")[:-3]
        session = (event.session_id or "-")[:8]
        turn = event.turn_id or "-"
        stage = event.stage or "-"
        name = event.name[len(stage) + 1:] if event.name.startswith(stage + ".") else event.name

        if event.name == "turn.summary":
            return f"{clock}  {session:<8} {turn:<4} {_summary_line(event)}"
        if event.name in ("model.downloading", "model.downloaded"):
            return f"{clock}  {_download_line(event)}"

        parts = []
        if event.level in ("warning", "error"):
            parts.append(event.level.upper())
        if event.duration_ms is not None:
            parts.append(f"{event.duration_ms:.0f}ms")
        attrs = dict(event.attrs)
        for key in _LEADING_ATTRS:
            if key in attrs:
                parts.append(f"{key}={_fmt_value(attrs.pop(key))}")
        parts.extend(f"{k}={_fmt_value(v)}" for k, v in sorted(attrs.items()) if not isinstance(v, (list, dict)))
        if event.request_id:
            parts.append(f"req={event.request_id[:8]}")
        line = f"{clock}  {session:<8} {turn:<4} {stage:<8} {name:<18} {' '.join(parts)}".rstrip()

        if event.error is not None:
            err = event.error
            line += f"\n{'':>14}{err.code} ({err.type}): {err.message}"
            line += f"  retryable={'yes' if err.retryable else 'no'}"
            if err.fix:
                line += f"\n{'':>14}→ {err.fix}"
            if err.stack:
                line += "\n" + "\n".join(f"{'':>14}| {l}" for l in err.stack.rstrip().splitlines())
        return line


def _summary_line(event: Event) -> str:
    a = event.attrs

    def ms(key: str) -> str:
        return f"{a[key]:.0f}ms" if a.get(key) is not None else "n/a"

    pieces = [f"turn {event.turn_id} {a.get('outcome', '?')}"]
    if a.get("ttfa_ms") is not None:
        pieces.append(f"TTFA {ms('ttfa_ms')}")
    pieces.append(f"response {ms('response_ms')}")
    if a.get("end_of_turn_wait_ms") is not None:
        pieces.append(f"end-of-turn wait {ms('end_of_turn_wait_ms')}")
    if a.get("stt_windows"):
        pieces.append(f"stt {a['stt_windows']}× {ms('stt_transcribe_ms')}")
    if a.get("llm_first_token_ms") is not None:
        rate = f", {a['llm_tokens_per_second']:.0f} tok/s" if a.get("llm_tokens_per_second") else ""
        pieces.append(f"llm first {ms('llm_first_token_ms')}{rate}")
    if a.get("tts_first_chunk_ms") is not None:
        rtf = f", rtf {a['tts_rtf']:.2f}" if a.get("tts_rtf") is not None else ""
        pieces.append(f"tts first {ms('tts_first_chunk_ms')}{rtf}")
    if a.get("interrupted"):
        pieces.append(f"interrupted, stopped in {ms('interruption_stop_ms')}")
    return " · ".join(pieces)


class JsonSink(Sink):
    """One JSON object per line: easy for Loki, Datadog, CloudWatch, or `jq`."""

    def __init__(self, stream: TextIO, min_level: str = "info"):
        self.stream = stream
        self.min_level = min_level
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), default=str, separators=(",", ":"))
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


class ListSink(Sink):
    """Collects events in memory; for tests and benchmarks."""

    def __init__(self, min_level: str = "debug"):
        self.min_level = min_level
        self.events: list = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def named(self, name: str) -> list:
        return [e for e in self.events if e.name == name]
