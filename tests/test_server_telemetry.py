"""Server observability: sessions always end (and say why), real errors reach the client with a
code and a fix, and /metrics and /health work. Uses a fake orchestrator, so no models load."""
import time

import pytest
from fastapi.testclient import TestClient

import fusion_runtime.server as server
from fusion_runtime.config import DEVELOPMENT_CONFIG
from fusion_runtime.contract import AuthFailed
from fusion_runtime.telemetry import ListSink, telemetry


class FakeOrchestrator:
    fail_with = None  # exception to raise after the first audio arrives

    def __init__(self, config):
        self.config = config
        self.ready = True

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_metrics_summary(self):
        return {"count": 0}

    async def run_pipeline(self, audio_stream, system_prompt, on_event=None, barge_in=None, trace=None):
        async for chunk in audio_stream:
            trace.audio_received(len(chunk))
            if FakeOrchestrator.fail_with is not None:
                raise FakeOrchestrator.fail_with
            yield b"\x00\x00" * 160


@pytest.fixture
def events(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    FakeOrchestrator.fail_with = None
    sink = ListSink()
    telemetry.add_sink(sink)
    yield sink
    telemetry.remove_sink(sink)


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_disconnect_ends_the_session(events):
    """Waiting only on the sender left sessions running forever after the client left."""
    with TestClient(server.app) as client:
        with client.websocket_connect("/v1/voice/ws") as ws:
            config = ws.receive_json()
            ws.send_bytes(b"\x00\x00" * 320)
            ws.receive_bytes()
        assert wait_for(lambda: events.named("session.end")), "session never ended after the client disconnected"
        end = events.named("session.end")[0]
        assert end.attrs["reason"] == "client_disconnected"
        assert end.session_id == config["session_id"]
        assert end.attrs["duration_s"] >= 0
        assert server._active_sessions == set()


def test_pipeline_error_reaches_client_with_code_and_fix(events):
    FakeOrchestrator.fail_with = AuthFailed("401 bad key sk-abcdefghijklmnop1234")
    with TestClient(server.app) as client:
        with client.websocket_connect("/v1/voice/ws") as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00\x00" * 320)
            error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "auth_failed"
        assert "Check the API key" in error["fix"]
        assert "sk-abc" not in error["message"], "secrets must not reach the client"
        assert "stack" not in error
        assert wait_for(lambda: events.named("session.end"))
        assert events.named("session.end")[0].attrs["reason"] == "error"
        assert events.named("session.error")[0].error.code == "auth_failed"


def test_runtime_errors_are_no_longer_swallowed_as_disconnects(events):
    FakeOrchestrator.fail_with = RuntimeError("decode failed: CUDA error")
    with TestClient(server.app) as client:
        with client.websocket_connect("/v1/voice/ws") as ws:
            ws.receive_json()
            ws.send_bytes(b"\x00\x00" * 320)
            error = ws.receive_json()
        assert error["type"] == "error" and error["code"] == "internal_error"
        assert wait_for(lambda: events.named("session.end"))
        assert events.named("session.end")[0].attrs["reason"] == "error"


def test_session_start_and_server_lifecycle_events(events):
    with TestClient(server.app) as client:
        with client.websocket_connect("/v1/voice/ws") as ws:
            ws.receive_json()
        assert wait_for(lambda: events.named("session.end"))
    names = [e.name for e in events.events]
    assert "server.start" in names and "server.ready" in names and "session.start" in names
    assert "server.stop" in names


def test_metrics_endpoint_is_prometheus_text(events):
    with TestClient(server.app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "fusion_sessions_active" in response.text
    assert "fusion_turn_response_seconds_bucket" in response.text
    assert "process_resident_memory_bytes" in response.text


def test_health_reports_version_uptime_and_sessions(events):
    with TestClient(server.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["version"] and body["uptime_s"] >= 0 and body["active_sessions"] == 0


class TalkingOrchestrator(FakeOrchestrator):
    """A pipeline that reports one finished turn, the way the real one does."""

    async def run_pipeline(self, audio_stream, system_prompt, on_event=None, barge_in=None, trace=None):
        if trace.on_turn_trace is None:  # the real pipeline wires traces to on_event this way
            trace.on_turn_trace = lambda turn_trace: on_event({"type": "turn.trace", **turn_trace})
        async for chunk in audio_stream:
            trace.audio_received(len(chunk))
        turn = trace.start_responding()
        on_event({"type": "transcript", "text": "I want to check my order.", "is_final": True})
        on_event({"type": "response", "text": "Sure, what is the order number?", "is_final": True})
        yield b"\x00\x00" * 160
        trace.end_turn(turn, "completed")  # sends the turn trace through on_event


def test_chat_returns_what_was_said_with_each_turn_s_metrics(monkeypatch):
    import base64

    # the app's startup builds the orchestrator, so replace the class it builds
    monkeypatch.setattr(server, "PipelineOrchestrator", TalkingOrchestrator)
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    with TestClient(server.app) as client:
        response = client.post("/v1/voice/chat", json={"audio_base64": base64.b64encode(b"\x00\x00" * 8000).decode()})
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "I want to check my order."
    assert body["response_text"] == "Sure, what is the order number?"
    assert base64.b64decode(body["audio_base64"])
    assert len(body["turns"]) == 1
    turn = body["turns"][0]
    assert turn["user"] == "I want to check my order." and turn["agent"] == "Sure, what is the order number?"
    assert turn["outcome"] == "completed" and turn["metrics"]["outcome"] == "completed"


def test_a_turn_discarded_as_the_agents_own_echo_is_not_reported_as_an_exchange():
    turns = []
    collect = server._collect_turns(turns)
    collect({"type": "transcript", "text": "the agent's own words", "is_final": True})
    collect({"type": "echo_discarded", "text": "the agent's own words"})
    collect({"type": "turn.trace", "summary": {"outcome": "echo_discarded"}})
    collect({"type": "transcript", "text": "a real question", "is_final": True})
    collect({"type": "response", "text": "a real answer", "is_final": True})
    collect({"type": "turn.trace", "summary": {"outcome": "completed", "ttfa_ms": 900}})
    assert [(t.user, t.agent) for t in turns] == [("a real question", "a real answer")]
    assert turns[0].metrics["ttfa_ms"] == 900
