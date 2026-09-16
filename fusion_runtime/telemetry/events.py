"""The telemetry event: one record of something that happened, with enough context to debug it."""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


@dataclass
class ErrorInfo:
    code: str  # stable, snake_case: "model_not_found", "auth_failed", "internal_error", ...
    type: str  # exception class name
    message: str  # secrets already redacted
    retryable: bool = False
    fix: Optional[str] = None  # what the user can do about it
    stage: Optional[str] = None
    stack: Optional[str] = None  # server-side only; never sent to clients

    def for_client(self) -> Dict[str, Any]:
        # "error_type", not "type": client messages already use "type" for the message kind
        return {k: v for k, v in {
            "code": self.code, "error_type": self.type, "message": self.message,
            "retryable": self.retryable, "fix": self.fix, "stage": self.stage,
        }.items() if v is not None}


@dataclass
class Event:
    name: str  # dotted: "llm.first_token", "session.start", "turn.summary"
    level: str = "info"
    ts: float = field(default_factory=time.time)  # wall clock, seconds since epoch
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    request_id: Optional[str] = None
    stage: Optional[str] = None  # "vad" | "stt" | "turn" | "llm" | "tts" | "audio" | "barge_in" | "server" | ...
    duration_ms: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorInfo] = None

    @property
    def iso_time(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat(timespec="milliseconds")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"ts": self.iso_time, "unix_ts": round(self.ts, 6), "level": self.level, "event": self.name}
        for key in ("session_id", "turn_id", "request_id", "stage"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.duration_ms is not None:
            out["duration_ms"] = round(self.duration_ms, 2)
        if self.attrs:
            out.update({k: v for k, v in self.attrs.items() if k not in out})
        if self.error is not None:
            out["error"] = {k: v for k, v in self.error.__dict__.items() if v is not None}
        return out
