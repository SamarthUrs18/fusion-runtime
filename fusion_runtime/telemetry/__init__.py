"""Telemetry: structured events, per-turn timelines, Prometheus metrics, and error context.

    from fusion_runtime.telemetry import telemetry
    telemetry.emit("llm.first_token", stage="llm", duration_ms=115, model="...")

Conversation text stays out of logs unless `log_content` is enabled
(`frun up --log-content`); secrets are always redacted.
"""
from fusion_runtime.telemetry.errors import describe_error, tag_stage
from fusion_runtime.telemetry.events import ErrorInfo, Event
from fusion_runtime.telemetry.hub import (
    Sink,
    Telemetry,
    current_session_id,
    session_scope,
    telemetry,
)
from fusion_runtime.telemetry.loop_monitor import LoopMonitor
from fusion_runtime.telemetry.sinks import ConsoleSink, JsonSink, ListSink
from fusion_runtime.telemetry.trace import SessionTrace, TurnTrace
