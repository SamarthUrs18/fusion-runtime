"""Prometheus metrics, fed by telemetry events and served at GET /metrics.

Metrics use their own registry (not prometheus_client's global one), so tests
and multiple servers in one process don't collide. Durations are in seconds,
as Prometheus conventions expect.
"""
import os
import sys
import time
from typing import Dict

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
LONG_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600)
RATE_BUCKETS = (1, 5, 10, 20, 40, 60, 100, 200, 500)
RTF_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0)

# turn.summary attribute (ms) -> histogram name suffix
_TURN_LATENCIES = {
    "ttfa_ms": ("turn_ttfa_seconds", "End of user speech to first reply audio sent (real-time audio only)"),
    "response_ms": ("turn_response_seconds", "Turn end detected to first reply audio sent"),
    "end_of_turn_wait_ms": ("turn_end_of_turn_wait_seconds", "End of user speech to turn end detected"),
    "transcription_delay_ms": ("stt_transcription_delay_seconds", "End of user speech to last transcript update"),
    "llm_first_token_ms": ("llm_first_token_seconds", "LLM request to first token"),
    "llm_duration_ms": ("llm_duration_seconds", "LLM request to last token"),
    "tts_first_chunk_ms": ("tts_first_chunk_seconds", "First LLM token to first synthesized audio"),
    "interruption_stop_ms": ("interruption_stop_seconds", "Barge-in detected to reply generation stopped"),
    "playback_delay_ms": ("playback_delay_seconds", "First audio sent to client reporting playback started"),
}


class _ResourceCollector:
    """Process and GPU numbers, read at scrape time."""

    def __init__(self) -> None:
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None

    def collect(self):
        if self._process is not None:
            try:
                with self._process.oneshot():
                    memory = self._process.memory_info()
                    cpu = self._process.cpu_times()
                    threads = self._process.num_threads()
                yield GaugeMetricFamily("process_resident_memory_bytes", "Resident memory", value=memory.rss)
                yield CounterMetricFamily("process_cpu_seconds", "User + system CPU time", value=cpu.user + cpu.system)
                yield GaugeMetricFamily("process_threads", "OS threads", value=threads)
            except Exception:
                pass
        torch = sys.modules.get("torch")  # never import torch just to scrape
        if torch is not None:
            try:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    used = GaugeMetricFamily("fusion_gpu_memory_allocated_bytes", "GPU memory allocated by torch",
                                             labels=["device"])
                    for index in range(torch.cuda.device_count()):
                        used.add_metric([str(index)], torch.cuda.memory_allocated(index))
                    yield used
            except Exception:
                pass


class TelemetryMetrics:
    min_level = "debug"

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.started_at = time.time()

        self.build_info = Gauge("fusion_build_info", "Version and profile of the running server", ["version", "profile"], registry=r)
        self.uptime = Gauge("fusion_uptime_seconds", "Seconds since the telemetry hub started", registry=r)
        self.uptime.set_function(lambda: time.time() - self.started_at)
        self.model_info = Gauge("fusion_model_info", "Loaded models", ["stage", "runtime", "model"], registry=r)
        self.model_load = Gauge("fusion_model_load_seconds", "Time to load and warm up a model", ["stage"], registry=r)

        self.sessions_active = Gauge("fusion_sessions_active", "Conversations in progress", registry=r)
        self.sessions_started = Counter("fusion_sessions_started", "Conversations started", registry=r)
        self.sessions_ended = Counter("fusion_sessions_ended", "Conversations ended", ["reason"], registry=r)
        self.session_duration = Histogram("fusion_session_duration_seconds", "Conversation length",
                                          buckets=LONG_BUCKETS, registry=r)

        self.turns = Counter("fusion_turns", "Turns handled", ["outcome"], registry=r)
        self.turn_latency: Dict[str, Histogram] = {
            key: Histogram(f"fusion_{name}", help_text, buckets=LATENCY_BUCKETS, registry=r)
            for key, (name, help_text) in _TURN_LATENCIES.items()
        }
        self.llm_tokens_per_second = Histogram("fusion_llm_tokens_per_second", "LLM generation speed",
                                               buckets=RATE_BUCKETS, registry=r)
        self.llm_tokens = Counter("fusion_llm_output_tokens", "LLM tokens generated", registry=r)
        self.tts_rtf = Histogram("fusion_tts_realtime_factor", "Synthesis time / audio time (below 1 is faster than real time)",
                                 buckets=RTF_BUCKETS, registry=r)
        self.tts_audio = Counter("fusion_tts_audio_seconds", "Seconds of speech synthesized", registry=r)
        self.stt_audio = Counter("fusion_stt_speech_seconds", "Seconds of user speech detected", registry=r)

        self.interruptions = Counter("fusion_interruptions", "Barge-ins that stopped a reply", registry=r)
        self.echo_discarded = Counter("fusion_echo_discarded", "User turns discarded as the bot's own echo", registry=r)
        self.scheduler_wait = Histogram("fusion_scheduler_wait_seconds", "Time a request waited for a model slot",
                                        ["stage"], buckets=LATENCY_BUCKETS, registry=r)
        self.scheduler_rejected = Counter("fusion_scheduler_rejected", "Requests refused because a model was at capacity",
                                          ["stage"], registry=r)
        self.errors = Counter("fusion_errors", "Errors by stage and code", ["stage", "code"], registry=r)
        self.loop_lag = Gauge("fusion_event_loop_lag_seconds", "Worst event-loop delay in the last window", registry=r)
        self.loop_stalls = Counter("fusion_event_loop_stalls", "Times the event loop was blocked over the stall threshold",
                                   registry=r)
        r.register(_ResourceCollector())

    # ---- events -> metrics ------------------------------------------------------------

    def handle(self, event) -> None:
        name, a = event.name, event.attrs
        if event.error is not None and event.level == "error":
            self.errors.labels(event.error.stage or event.stage or "unknown", event.error.code).inc()

        if name == "server.start":
            self.build_info.labels(a.get("version", "unknown"), a.get("profile", "unknown")).set(1)
        elif name == "model.loaded":
            self.model_info.labels(event.stage or "?", a.get("runtime", "?"), a.get("model", "?")).set(1)
            if event.duration_ms is not None:
                self.model_load.labels(event.stage or "?").set(event.duration_ms / 1000)
        elif name == "session.start":
            self.sessions_active.inc()
            self.sessions_started.inc()
        elif name == "session.end":
            self.sessions_active.dec()
            self.sessions_ended.labels(a.get("reason", "unknown")).inc()
            if a.get("duration_s") is not None:
                self.session_duration.observe(a["duration_s"])
        elif name == "turn.summary":
            self.turns.labels(a.get("outcome", "unknown")).inc()
            for key, histogram in self.turn_latency.items():
                if a.get(key) is not None:
                    histogram.observe(a[key] / 1000)
            if a.get("llm_tokens_per_second"):
                self.llm_tokens_per_second.observe(a["llm_tokens_per_second"])
            if a.get("llm_tokens"):
                self.llm_tokens.inc(a["llm_tokens"])
            if a.get("tts_rtf") is not None:
                self.tts_rtf.observe(a["tts_rtf"])
            if a.get("tts_audio_s"):
                self.tts_audio.inc(a["tts_audio_s"])
            if a.get("speech_ms"):
                self.stt_audio.inc(a["speech_ms"] / 1000)
        elif name == "scheduler.slot":
            if event.duration_ms is not None:
                self.scheduler_wait.labels(event.stage or "?").observe(event.duration_ms / 1000)
        elif name == "scheduler.rejected":
            self.scheduler_rejected.labels(event.stage or "?").inc()
        elif name == "barge_in.fired":
            self.interruptions.inc()
        elif name == "echo.discarded":
            self.echo_discarded.inc()
        elif name == "event_loop.lag":
            self.loop_lag.set(a.get("max_lag_ms", 0) / 1000)
        elif name == "event_loop.stall":
            self.loop_stalls.inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)

    content_type = CONTENT_TYPE_LATEST
