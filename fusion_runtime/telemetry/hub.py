"""The telemetry hub: every component emits events here; sinks decide where they go.

Sinks: a human console, JSON lines for log collectors, Prometheus metrics, and
per-session subscribers (the WebSocket layer, to send traces and errors to
clients). A failing sink never breaks the pipeline.
"""
import contextlib
import contextvars
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO

from fusion_runtime.telemetry.events import LEVELS, ErrorInfo, Event
from fusion_runtime.telemetry.redact import redact_secrets

_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("fusion_session_id", default=None)


def current_session_id() -> Optional[str]:
    return _session_id.get()


@contextlib.contextmanager
def session_scope(session_id: str) -> Iterator[None]:
    """Everything emitted inside (including tasks created inside) carries this session id."""
    token = _session_id.set(session_id)
    try:
        yield
    finally:
        _session_id.reset(token)


_stderr_copy: Optional[TextIO] = None


def _private_stderr() -> TextIO:
    """Our own handle on the process's stderr.

    Native libraries silence output by pointing file descriptor 2 at /dev/null
    for a while (llama.cpp does this while loading a model). Anything written to
    sys.stderr meanwhile is lost, including our logs. A duplicate of the original
    descriptor keeps pointing at the real destination, so no event disappears.
    """
    global _stderr_copy
    if _stderr_copy is None:
        try:
            _stderr_copy = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1, encoding="utf-8")
        except (AttributeError, OSError, ValueError):  # stderr replaced (e.g. by a test runner): use it as is
            return sys.stderr
    return _stderr_copy


class Sink:
    min_level: str = "debug"

    def handle(self, event: Event) -> None:
        raise NotImplementedError


class Telemetry:
    def __init__(self) -> None:
        from fusion_runtime.telemetry.metrics import TelemetryMetrics

        self.log_content = False
        self.metrics = TelemetryMetrics()
        self._log_sinks: List[Sink] = []
        self._extra_sinks: List[Sink] = []
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._configured = False
        self.sink_failures = 0

    # ---- configuration ----------------------------------------------------------

    def configure(
        self,
        *,
        format: str = "pretty",  # "pretty" | "json" | "off"
        level: str = "info",
        log_content: bool = False,
        stream: Optional[TextIO] = None,
    ) -> None:
        from fusion_runtime.telemetry.sinks import ConsoleSink, JsonSink

        if format not in ("pretty", "json", "off"):
            raise ValueError(f"log format must be pretty, json or off, got {format!r}")
        if level not in LEVELS:
            raise ValueError(f"log level must be one of {', '.join(LEVELS)}, got {level!r}")
        stream = stream if stream is not None else _private_stderr()
        sinks: List[Sink] = []
        if format == "pretty":
            sinks.append(ConsoleSink(stream, level))
        elif format == "json":
            sinks.append(JsonSink(stream, level))
        with self._lock:
            self._log_sinks = sinks
            self.log_content = log_content
            self._configured = True

    def configure_from_env(self) -> None:
        """FUSION_LOG_FORMAT (pretty|json|off), FUSION_LOG_LEVEL, FUSION_LOG_CONTENT (1 to include text)."""
        self.configure(
            format=os.getenv("FUSION_LOG_FORMAT", "pretty"),
            level=os.getenv("FUSION_LOG_LEVEL", "info"),
            log_content=os.getenv("FUSION_LOG_CONTENT", "0").lower() in ("1", "true", "yes"),
        )

    def add_sink(self, sink: Sink) -> None:
        with self._lock:
            self._extra_sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        with self._lock:
            if sink in self._extra_sinks:
                self._extra_sinks.remove(sink)

    def subscribe(self, session_id: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(session_id, []).append(callback)

    def unsubscribe(self, session_id: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            callbacks = self._subscribers.get(session_id, [])
            if callback in callbacks:
                callbacks.remove(callback)
            if not callbacks:
                self._subscribers.pop(session_id, None)

    # ---- emitting ---------------------------------------------------------------

    def content(self, text: Optional[str], key: str = "text") -> Dict[str, Any]:
        """Attributes describing conversation text: always its length, the text itself only if opted in."""
        text = text or ""
        attrs: Dict[str, Any] = {f"{key}_chars": len(text)}
        if self.log_content:
            attrs[key] = text
        return attrs

    def emit(
        self,
        name: str,
        *,
        level: str = "info",
        stage: Optional[str] = None,
        session_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[ErrorInfo] = None,
        **attrs: Any,
    ) -> Event:
        if not self._configured:
            self.configure()  # library use without explicit setup: readable console at info
        event = Event(
            name=name,
            level=level,
            session_id=session_id or current_session_id(),
            turn_id=turn_id,
            request_id=request_id,
            stage=stage or name.split(".", 1)[0],
            duration_ms=duration_ms,
            attrs={k: redact_secrets(v) for k, v in attrs.items() if v is not None},
            error=error,
        )
        with self._lock:
            sinks = [self.metrics, *self._log_sinks, *self._extra_sinks]
            subscribers = list(self._subscribers.get(event.session_id or "", []))
        for sink in sinks:
            if LEVELS[event.level] < LEVELS[getattr(sink, "min_level", "debug")]:
                continue
            try:
                sink.handle(event)
            except Exception:
                self.sink_failures += 1
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                self.sink_failures += 1
        return event


telemetry = Telemetry()
