"""Nothing heavy may run on the event loop: it freezes audio input and barge-in for every conversation.

Each test here pins one blocking call that telemetry's event-loop monitor caught in a real
session (a 192 ms stall from Whisper decoding, 50-80 ms from Silero loads and inference).
"""
import asyncio
import threading

import numpy as np
import pytest

from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import ModelSpec, STTRequest, Transcript
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.runtimes.ctranslate2.stt import CTranslate2STT
from fusion_runtime.telemetry import ListSink, telemetry

MAIN = threading.main_thread()


class _Segment:
    def __init__(self, text):
        self.text = text


class _Info:
    language = "en"
    language_probability = 1.0


class LazyWhisperModel:
    """Like faster-whisper: transcribe() is cheap, iterating the segments does the decoding."""

    def __init__(self):
        self.decode_threads = []

    def transcribe(self, audio, **options):
        def segments():
            self.decode_threads.append(threading.current_thread())
            yield _Segment(" hello")
            yield _Segment(" there")
        return segments(), _Info()


async def test_whisper_decoding_happens_off_the_event_loop():
    stt = CTranslate2STT(ModelSpec(stage="stt", runtime="ctranslate2", model="whisper"))
    stt.model = LazyWhisperModel()

    results = await stt.transcribe([STTRequest(audio=b"\x00\x00" * 16000)])
    assert isinstance(results[0], Transcript) and results[0].text == " hello  there"
    assert stt.model.decode_threads, "segments were never decoded"
    assert MAIN not in stt.model.decode_threads, "Whisper segments were decoded on the event loop thread"


def _orchestrator_with_fake_vad(model):
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    vad = type("FakeVAD", (), {})()
    vad.config = type("C", (), {"threshold": 0.5})()
    vad.sample_rate = 16000
    vad._frame_model = model
    orch.vad = vad
    return orch


async def test_vad_inference_runs_off_the_event_loop():
    threads = []

    def fake_silero(tensor, sr):
        import torch
        threads.append(threading.current_thread())
        return torch.tensor(float(tensor.abs().mean() > 0.1))

    orch = _orchestrator_with_fake_vad(fake_silero)

    async def audio():
        for _ in range(20):
            yield np.full(512, 10000, dtype=np.int16).tobytes()

    kept = [frame async for frame in orch._apply_vad(audio())]
    assert len(kept) == 20
    assert threads and MAIN not in threads, "Silero ran on the event loop thread"
    orch.__dict__.pop("_vad_pool").shutdown(wait=True)


class CopyableModel:
    loads = 0

    def __init__(self):
        CopyableModel.loads += 1
        self.state = 0
        self.resets = 0

    def reset_states(self):
        self.state = 0
        self.resets += 1


async def test_each_stream_gets_its_own_fresh_copy_of_one_loaded_model(monkeypatch):
    orch = _orchestrator_with_fake_vad(None)
    template = CopyableModel()
    template.state = 99  # the template has been used

    async def load_template():
        return template

    monkeypatch.setattr(orch, "_load_vad_template", load_template)
    first, second = await asyncio.gather(orch._load_vad_frame_model(), orch._load_vad_frame_model())
    assert first is not second and first is not template and second is not template
    assert first.state == second.state == 0, "copies must start with fresh state"
    assert CopyableModel.loads == 1, "the model must be loaded once, not per stream"


async def test_failed_vad_load_is_reported_not_silent(monkeypatch):
    import torch

    def broken_load(*args, **kwargs):
        raise OSError("dlopen: Symbol not found (torchaudio ABI mismatch)")

    monkeypatch.setattr(torch.hub, "load", broken_load)
    sink = ListSink()
    telemetry.add_sink(sink)
    try:
        orch = _orchestrator_with_fake_vad(None)
        assert await orch._load_vad_frame_model() is None
    finally:
        telemetry.remove_sink(sink)
    failed = sink.named("model.load_failed")
    assert failed and failed[0].stage == "vad" and failed[0].level == "warning"
    assert "interruptions don't work" in failed[0].attrs["impact"]
