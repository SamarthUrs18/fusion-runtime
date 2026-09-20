"""
Tests for real-interruption ("barge-in") support: the independent audio
watcher that can cancel an in-flight LLM/TTS reply when it detects genuine,
sustained user speech — see PipelineOrchestrator._barge_in_watcher.

These use a fake, energy-based "VAD model" (loud = speech, silent = not)
instead of the real Silero weights, so they're fast and deterministic and
don't depend on model-cache/network state. Full end-to-end behavior with
real STT/LLM/TTS still needs to be exercised live (see examples/websocket_client.py).
"""
import asyncio

import numpy as np
import pytest
from fusion_runtime.config import PipelineConfig
from fusion_runtime.engine import BargeInState, PipelineOrchestrator


def make_orchestrator(threshold: float = 0.5, sample_rate: int = 16000, min_speech_ms: int = 300):
    """A PipelineOrchestrator with only the config + a fake VAD frame model
    set up — no real STT/LLM/TTS/VAD factories are touched."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.turn_detection.barge_in_min_speech_ms = min_speech_ms

    class _Config:
        pass

    fake_vad = type("FakeVAD", (), {})()
    fake_vad.config = _Config()
    fake_vad.config.threshold = threshold
    fake_vad.sample_rate = sample_rate

    def fake_model(tensor, sr):
        # Stand-in for Silero: loud (>0.1 mean abs amplitude) = speech.
        # Avoids depending on real model weights/cache for a unit test.
        import torch
        return torch.tensor(float(tensor.abs().mean() > 0.1))

    fake_vad._frame_model = fake_model
    orch.vad = fake_vad
    return orch


def speech_frame(n_samples: int = 512, amplitude: int = 10000) -> bytes:
    return np.full(n_samples, amplitude, dtype=np.int16).tobytes()


def silence_frame(n_samples: int = 512) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


class TestBargeInWatcher:
    async def test_ignores_speech_when_bot_not_speaking(self):
        orch = make_orchestrator()
        barge_in = BargeInState()  # speaking=False by default — nothing to interrupt

        async def audio():
            for _ in range(50):  # far more than min_speech_ms worth of frames
                yield speech_frame()

        await orch._barge_in_watcher(audio(), barge_in, emit=None)
        assert not barge_in.interrupted.is_set()

    async def test_detects_sustained_speech_while_bot_speaking(self):
        orch = make_orchestrator(min_speech_ms=300)
        barge_in = BargeInState()
        barge_in.mark_speaking()
        events = []

        async def audio():
            for _ in range(30):  # 30 * 32ms = 960ms, well past the 300ms bar
                yield speech_frame()

        await orch._barge_in_watcher(audio(), barge_in, emit=events.append)
        assert barge_in.interrupted.is_set()
        assert {"type": "interrupted"} in events
        assert barge_in.speaking is False, "should stop watching until the next turn starts"

    async def test_still_interruptible_while_the_client_plays_after_generation_ends(self):
        """Replies are generated faster than they're spoken, so the server
        usually finishes generating seconds before the client's speaker
        finishes. The user must still be able to interrupt that tail."""
        orch = make_orchestrator(min_speech_ms=300)
        barge_in = BargeInState()
        barge_in.mark_speaking()
        barge_in.mark_idle()        # server done generating...
        barge_in.set_playing(True)  # ...client still playing it
        events = []

        async def audio():
            for _ in range(30):
                yield speech_frame()

        await orch._barge_in_watcher(audio(), barge_in, emit=events.append)
        assert barge_in.interrupted.is_set()
        assert {"type": "interrupted"} in events

    async def test_ignores_brief_blips(self):
        orch = make_orchestrator(min_speech_ms=300)
        barge_in = BargeInState()
        barge_in.mark_speaking()

        async def audio():
            # ~96ms of "speech" (well under the 300ms sustained-speech bar),
            # e.g. a cough or a residual echo fragment, then silence.
            for _ in range(3):
                yield speech_frame()
            for _ in range(20):
                yield silence_frame()

        await orch._barge_in_watcher(audio(), barge_in, emit=None)
        assert not barge_in.interrupted.is_set()

    async def test_mark_idle_resets_state(self):
        barge_in = BargeInState()
        barge_in.mark_speaking()
        barge_in.interrupted.set()
        barge_in.mark_idle()
        assert barge_in.speaking is False
        assert not barge_in.interrupted.is_set()


class TestTeeAudio:
    async def test_fans_out_to_all_queues_and_terminates(self):
        async def source():
            for i in range(5):
                yield bytes([i])

        q1: asyncio.Queue = asyncio.Queue()
        q2: asyncio.Queue = asyncio.Queue()
        await PipelineOrchestrator._tee_audio(source(), [q1, q2])

        got1 = [item async for item in PipelineOrchestrator._drain(q1)]
        got2 = [item async for item in PipelineOrchestrator._drain(q2)]
        expected = [bytes([i]) for i in range(5)]
        assert got1 == expected
        assert got2 == expected

    async def test_drain_stops_on_empty_source(self):
        async def empty_source():
            return
            yield  # pragma: no cover - makes this an async generator

        q: asyncio.Queue = asyncio.Queue()
        await PipelineOrchestrator._tee_audio(empty_source(), [q])
        got = [item async for item in PipelineOrchestrator._drain(q)]
        assert got == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
