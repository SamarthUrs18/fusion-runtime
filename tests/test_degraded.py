"""A server running below strength says so for as long as it runs, not only once at start-up.

The VAD was once broken for several sessions unnoticed: its load failure was a single line at start.
"""
from fusion_runtime import server
from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import Health
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.runtimes.onnx.tts import OnnxTTS
from fusion_runtime.telemetry.events import Event
from fusion_runtime.telemetry.metrics import TelemetryMetrics
from starlette.testclient import TestClient


class Runtime:
    def __init__(self, health):
        self._health = health

    def health(self):
        return self._health


def orchestrator(**runtimes):
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    for stage, runtime in runtimes.items():
        setattr(orch, stage, runtime)
    return orch


def test_degraded_parts_are_collected_from_the_vad_and_each_runtime():
    orch = orchestrator(stt=Runtime(Health("ok")), tts=Runtime(Health("degraded", "running on the CPU")))
    orch.__dict__["_degraded"] = {"vad": "not loaded: turns end on a timer"}
    assert orch.degraded() == {"vad": "not loaded: turns end on a timer", "tts": "running on the CPU"}
    assert orchestrator(stt=Runtime(Health("ok"))).degraded() == {}


def test_kokoro_on_the_cpu_reports_itself_degraded():
    tts = OnnxTTS.__new__(OnnxTTS)
    tts._loaded = True

    class Family:
        degraded = "text-to-speech is running on the CPU: CUDAExecutionProvider didn't load"

    tts.family = Family()
    assert tts.health().status == "degraded" and "CPU" in tts.health().detail
    Family.degraded = None
    assert tts.health().status == "ok"


def test_health_and_every_session_say_so(monkeypatch):
    from fusion_runtime.telemetry import telemetry
    from fusion_runtime.telemetry.sinks import ListSink

    class DegradedOrchestrator:
        def __init__(self, config):
            self.config, self.ready = config, True

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

        def degraded(self):
            return {"vad": "not loaded: turns end on a timer"}

        async def run_pipeline(self, audio_stream, system_prompt, **kwargs):
            async for _ in audio_stream:
                if False:
                    yield b""

    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setenv("FUSION_ACCEPTED_KEYS", "")
    monkeypatch.setattr(server, "PipelineOrchestrator", DegradedOrchestrator)
    sink = ListSink()
    with TestClient(server.app, client=("127.0.0.1", 4000)) as client:
        telemetry.add_sink(sink)
        try:
            health = client.get("/health").json()
            assert health["status"] == "degraded" and "vad" in health["degraded"]
            with client.websocket_connect("/v1/voice/ws") as ws:
                ws.receive_json()
        finally:
            telemetry.remove_sink(sink)
    start = sink.named("session.start")[0]
    assert start.level == "warning" and start.attrs["degraded"] == "vad"


def test_degraded_parts_become_a_gauge():
    from prometheus_client import generate_latest

    metrics = TelemetryMetrics()
    metrics.handle(Event(name="server.degraded", stage="server", attrs={"components": ["vad", "tts"]}))
    text = generate_latest(metrics.registry).decode()
    assert 'fusion_degraded{component="vad"} 1.0' in text and 'fusion_degraded{component="tts"} 1.0' in text
