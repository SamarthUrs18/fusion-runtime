"""Telemetry: events carry ids and timestamps, secrets and content stay out of logs,
errors carry context and a fix, metrics count what happened, traces time each turn."""
import asyncio
import io
import json
import re
import time

import numpy as np
import pytest

from fusion_runtime.contract import AuthFailed, RateLimited
from fusion_runtime.telemetry import (
    ConsoleSink,
    ListSink,
    LoopMonitor,
    SessionTrace,
    Telemetry,
    describe_error,
    session_scope,
    tag_stage,
)
from fusion_runtime.telemetry.metrics import TelemetryMetrics


@pytest.fixture
def hub():
    h = Telemetry()
    h.configure(format="off")
    sink = ListSink()
    h.add_sink(sink)
    h.sink = sink
    return h


# ---- events ---------------------------------------------------------------------------

def test_events_carry_session_id_timestamp_and_stage(hub):
    with session_scope("s-123"):
        event = hub.emit("llm.first_token", duration_ms=115.2, model="m")
    assert event.session_id == "s-123"
    assert event.stage == "llm"  # inferred from the name
    assert abs(event.ts - time.time()) < 5
    as_dict = event.to_dict()
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", as_dict["ts"])
    assert as_dict["duration_ms"] == 115.2 and as_dict["model"] == "m"


def test_session_scope_reaches_tasks_created_inside(hub):
    async def run():
        with session_scope("s-task"):
            return await asyncio.create_task(asyncio.sleep(0, result=hub.emit("x.y")))

    assert asyncio.run(run()).session_id == "s-task"


def test_content_is_left_out_unless_enabled(hub):
    assert hub.content("hello there") == {"text_chars": 11}
    hub.configure(format="off", log_content=True)
    assert hub.content("hello there", "reply") == {"reply_chars": 11, "reply": "hello there"}


def test_secrets_are_redacted_from_attrs_and_errors(hub):
    event = hub.emit("x.y", note="using sk-abcdefghijklmnop1234 now", header="Bearer abc.def.ghi12345")
    assert "sk-abc" not in event.attrs["note"] and "[REDACTED]" in event.attrs["note"]
    assert "abc.def" not in event.attrs["header"]
    info = describe_error(RuntimeError("401 for key gsk_abcdefghijklmnop123"))
    assert "gsk_" not in info.message


def test_json_lines_are_parseable_and_level_filtered():
    h, stream = Telemetry(), io.StringIO()
    h.configure(format="json", level="info", stream=stream)
    h.emit("stt.partial", level="debug", stage="stt")
    h.emit("stt.final", stage="stt", text_chars=12)
    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 1, "debug event should be filtered at info level"
    record = json.loads(lines[0])
    assert record["event"] == "stt.final" and record["level"] == "info" and "unix_ts" in record


def test_console_line_has_time_ids_and_error_fix_and_stack():
    h, stream = Telemetry(), io.StringIO()
    h.configure(format="pretty", level="debug", stream=stream)
    try:
        raise AuthFailed("401 invalid key")
    except AuthFailed as e:
        h.emit("llm.error", level="error", session_id="abcdef123456", turn_id="t3", error=describe_error(tag_stage(e, "llm")))
    out = stream.getvalue()
    assert re.match(r"\d\d:\d\d:\d\d\.\d{3}  abcdef12 t3", out)
    assert "auth_failed (AuthFailed): 401 invalid key" in out
    assert "→ Check the API key" in out
    assert "| Traceback" in out


def test_failing_sink_never_breaks_emit(hub):
    class Broken:
        min_level = "debug"

        def handle(self, event):
            raise RuntimeError("disk full")

    hub.add_sink(Broken())
    hub.emit("x.y")
    assert hub.sink_failures == 1
    assert hub.sink.named("x.y")


def test_subscribers_only_get_their_session(hub):
    got = []
    hub.subscribe("a", got.append)
    hub.emit("x.y", session_id="a")
    hub.emit("x.y", session_id="b")
    hub.unsubscribe("a", got.append)
    hub.emit("x.y", session_id="a")
    assert [e.session_id for e in got] == ["a"]


def test_configure_from_env(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "json")
    monkeypatch.setenv("FUSION_LOG_LEVEL", "debug")
    monkeypatch.setenv("FUSION_LOG_CONTENT", "1")
    h = Telemetry()
    h.configure_from_env()
    assert h.log_content is True
    with pytest.raises(ValueError):
        h.configure(format="xml")


# ---- errors -----------------------------------------------------------------------

def test_error_codes_and_fixes():
    missing = describe_error(FileNotFoundError("LLM model not found at /x/llm/m.gguf"))
    assert missing.code == "model_not_found" and "frun models pull" in missing.fix

    limited = describe_error(RateLimited("slow down"))
    assert limited.code == "rate_limited" and limited.retryable

    class RateLimitError(Exception):  # shaped like the OpenAI client's error, without importing it
        pass

    assert describe_error(RateLimitError("429")).code == "rate_limited"
    unknown = describe_error(ValueError("weird"))
    assert unknown.code == "internal_error" and unknown.fix
    assert describe_error(tag_stage(ValueError("x"), "tts")).stage == "tts"


def test_client_view_of_an_error_has_no_stack():
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        info = describe_error(e, "llm")
    assert info.stack and "stack" not in info.for_client()


# ---- metrics ----------------------------------------------------------------------

def test_metrics_follow_events():
    m, h = TelemetryMetrics(), Telemetry()
    h.metrics = m
    h.configure(format="off")
    h.emit("session.start", stage="server")
    h.emit("turn.summary", stage="turn", outcome="completed", response_ms=512.0, llm_first_token_ms=120.0,
           llm_tokens=40, llm_tokens_per_second=55.0, tts_rtf=0.25, tts_audio_s=3.5)
    h.emit("llm.error", level="error", error=describe_error(tag_stage(AuthFailed("x"), "llm")))
    h.emit("barge_in.fired")
    h.emit("session.end", stage="server", reason="client_disconnected", duration_s=12.0)
    text = m.render().decode()
    assert 'fusion_turns_total{outcome="completed"} 1.0' in text
    assert "fusion_turn_response_seconds_count 1.0" in text
    assert 'fusion_errors_total{code="auth_failed",stage="llm"} 1.0' in text
    assert "fusion_interruptions_total 1.0" in text
    assert "fusion_sessions_active 0.0" in text
    assert "fusion_llm_output_tokens_total 40.0" in text
    assert "process_resident_memory_bytes" in text


# ---- per-turn trace ---------------------------------------------------------------------

def test_next_turn_can_start_while_bot_still_responding(hub):
    trace = SessionTrace(session_id="s", hub=hub)
    first = trace.listening_turn()
    responding = trace.start_responding()
    assert responding is first and trace.listening is None
    second = trace.listening_turn()  # user starts talking during the reply
    assert second.turn_id == "t2" and trace.responding is first


def test_turn_summary_and_trace(hub):
    traces = []
    trace = SessionTrace(session_id="s", hub=hub, on_turn_trace=traces.append)
    for _ in range(50):  # 1 s of audio arriving in real time (fake clock)
        trace.audio_received(640, now=time.monotonic())
    turn = trace.listening_turn()
    turn.mark("speech_start")
    turn.add("speech_audio_ms", 800)
    turn.mark("speech_end")
    turn.mark("turn_end_detected")
    trace.start_responding()
    turn.mark("llm_request")
    turn.mark("llm_first_token")
    turn.add("llm_tokens", 11)
    turn.add("llm_decode_ms", 200)  # 10 tokens after the first in 200 ms of waiting on the model
    turn.mark("llm_done")
    turn.add("tts_audio_s", 2.0)
    turn.add("tts_synth_ms", 500)
    turn.mark("audio_first_sent")
    trace.end_turn(turn, "completed")

    summary = hub.sink.named("turn.summary")[0].attrs
    assert summary["outcome"] == "completed"
    assert summary["llm_tokens_per_second"] == 50.0
    assert summary["tts_rtf"] == 0.25
    assert summary["speech_ms"] == 800
    assert "response_ms" in summary
    assert traces and traces[0]["turn_id"] == "t1"
    timeline = [step["event"] for step in traces[0]["timeline"]]
    assert timeline[0] == "speech_start" and timeline[-1] == "turn_end"
    assert trace.responding is None


def test_ttfa_is_omitted_for_audio_faster_than_real_time(hub):
    trace = SessionTrace(session_id="s", hub=hub)
    now = time.monotonic()
    trace.audio_received(16000 * 2 * 3, now=now)  # 3 s of audio arriving at once (a file)
    assert trace.realtime_audio is False
    turn = trace.listening_turn()
    turn.mark("speech_end")
    turn.mark("audio_first_sent")
    assert "ttfa_ms" not in trace.summarize(turn, "completed")


def test_arrival_time_lookup(hub):
    trace = SessionTrace(session_id="s", hub=hub)
    trace.audio_received(640, now=100.0)   # samples 0..319
    trace.audio_received(640, now=100.02)  # samples 320..639
    assert trace.arrival_time(10) == 100.0
    assert trace.arrival_time(400) == 100.02


def test_finish_closes_open_turns(hub):
    trace = SessionTrace(session_id="s", hub=hub)
    trace.listening_turn().mark("speech_start")
    trace.finish()
    assert hub.sink.named("turn.summary")[0].attrs["outcome"] == "no_transcript"


# ---- event loop monitor -----------------------------------------------------------------

async def test_loop_monitor_reports_blocking(hub):
    monitor = LoopMonitor(hub=hub, interval_s=0.02, stall_ms=100)
    monitor.start()
    await asyncio.sleep(0.05)
    time.sleep(0.25)  # blocking call on the event loop
    await asyncio.sleep(0.05)
    await monitor.stop()
    stalls = hub.sink.named("event_loop.stall")
    assert stalls and stalls[0].duration_ms >= 200
    assert monitor.max_lag_ms >= 200


# ---- barge-in and backlog -------------------------------------------------------------------

def _watcher_orchestrator(min_speech_ms=300):
    from fusion_runtime.config import PipelineConfig
    from fusion_runtime.engine import PipelineOrchestrator

    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.barge_in_min_speech_ms = min_speech_ms
    vad = type("FakeVAD", (), {})()
    vad.config = type("C", (), {"threshold": 0.5})()
    vad.sample_rate = 16000

    def fake_model(tensor, sr):
        import torch
        return torch.tensor(float(tensor.abs().mean() > 0.1))

    vad._frame_model = fake_model
    orch.vad = vad
    return orch


def _speech_frames(n):
    return [np.full(512, 10000, dtype=np.int16).tobytes() for _ in range(n)]


async def test_speech_that_arrived_before_the_bot_spoke_is_not_an_interruption(hub):
    """The watcher can lag (model loading, a file sent at once). The user's own words from
    before the reply must not count as talking over the bot."""
    from fusion_runtime.engine import BargeInState

    orch, trace, barge_in = _watcher_orchestrator(), SessionTrace(session_id="s", hub=hub), BargeInState()
    frames = _speech_frames(30)
    for frame in frames:  # all arrived...
        trace.audio_received(len(frame))
    await asyncio.sleep(0.01)
    barge_in.mark_speaking()  # ...before the bot started speaking

    async def audio():
        for frame in frames:
            yield frame

    await orch._barge_in_watcher(audio(), barge_in, emit=None, trace=trace)
    assert not barge_in.interrupted.is_set()


async def test_speech_arriving_after_the_bot_spoke_still_interrupts(hub):
    from fusion_runtime.engine import BargeInState

    orch, trace, barge_in = _watcher_orchestrator(), SessionTrace(session_id="s", hub=hub), BargeInState()
    trace.listening_turn()
    trace.start_responding()
    barge_in.mark_speaking()
    await asyncio.sleep(0.01)
    frames = _speech_frames(30)

    async def audio():
        for frame in frames:
            trace.audio_received(len(frame))  # arrives while the bot is speaking
            yield frame

    await orch._barge_in_watcher(audio(), barge_in, emit=None, trace=trace)
    assert barge_in.interrupted.is_set()
    fired = hub.sink.named("barge_in.fired")
    assert fired and fired[0].turn_id == "t1" and fired[0].attrs["speech_over_bot_ms"] >= 300


def test_logs_survive_native_libraries_silencing_stderr(tmp_path):
    """llama.cpp points fd 2 at /dev/null while loading a model; our log lines must still arrive."""
    import os
    import subprocess
    import sys

    script = (
        "import os, sys\n"
        "from fusion_runtime.telemetry import Telemetry\n"
        "h = Telemetry(); h.configure(format='json')\n"
        "devnull = os.open(os.devnull, os.O_WRONLY); saved = os.dup(2); os.dup2(devnull, 2)\n"
        "h.emit('model.loaded', stage='tts')\n"
        "os.dup2(saved, 2)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert '"event":"model.loaded"' in result.stderr


def test_transcription_delay_is_never_negative(hub):
    trace = SessionTrace(session_id="s", hub=hub)
    trace.audio_received(640)
    turn = trace.listening_turn()
    turn.mark("stt_last_partial")  # transcript already up to date...
    time.sleep(0.01)
    turn.mark("speech_end")  # ...before VAD marked speech as ended
    assert trace.summarize(turn, "completed")["transcription_delay_ms"] == 0


def test_downloads_read_as_sentences_not_attributes():
    """A download is the one thing that keeps someone waiting: say it plainly, twice."""
    from fusion_runtime.telemetry.events import Event
    from fusion_runtime.telemetry.sinks import ConsoleSink

    starting = ConsoleSink.format(Event(name="model.downloading", stage="stt",
                                        attrs={"model": "hf:org/model", "size": "486 MB"}))
    finished = ConsoleSink.format(Event(name="model.downloaded", stage="stt", duration_ms=131_000,
                                        attrs={"model": "hf:org/model", "size": "486 MB"}))
    assert "Downloading the speech-to-text model hf:org/model (486 MB)" in starting
    assert "first time only" in starting and "Please wait" in starting
    assert "Downloaded the speech-to-text model hf:org/model in 2m 11s" in finished
    assert "starts straight away" in finished
    # a download whose size the hub didn't report still says what it is doing
    assert "Downloading the speech-to-text model hf:org/model." in ConsoleSink.format(
        Event(name="model.downloading", stage="stt", attrs={"model": "hf:org/model"}))
