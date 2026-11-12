"""Pre-roll: the audio just before speech starts reaches the recogniser too.

Silero is only sure a frame or two into a word. Its soft first sounds (the "w" of "wait") scored
under the threshold and were dropped, so a short word said while cutting in lost its start.
"""
import numpy as np
import pytest
from fusion_runtime.agent import VAD, Agent, AgentError
from fusion_runtime.config import PipelineConfig
from fusion_runtime.engine import PipelineOrchestrator

FRAME = 512  # samples: Silero's 32 ms frame


def frame(n: int) -> bytes:
    """A frame whose samples all equal n, so the test can tell frames apart."""
    return np.full(FRAME, n, dtype=np.int16).tobytes()


def orchestrator(pre_roll_ms: int, probabilities):
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.config.vad.pre_roll_ms = pre_roll_ms
    vad = type("FakeVAD", (), {})()
    vad.config = type("C", (), {"threshold": 0.5})()
    vad.sample_rate = 16000
    orch.vad = vad

    async def model():
        return object()

    async def score(_model, frames, _rate):
        return [probabilities[int(np.frombuffer(f, dtype=np.int16)[0])] for f in frames]

    orch._load_vad_frame_model = model
    orch._vad_probabilities = score
    return orch


async def kept(orch, count):
    async def audio():
        for n in range(count):
            yield frame(n)

    return [int(np.frombuffer(f, dtype=np.int16)[0]) async for f in orch._apply_vad(audio())]


# frames 0-19 room tone, 20-21 a soft first sound under the threshold, 22-26 speech
ONSET = [0.05] * 20 + [0.3, 0.4] + [0.9] * 5


async def test_the_frames_before_speech_are_sent_with_it():
    orch = orchestrator(256, ONSET)  # 8 frames
    assert await kept(orch, len(ONSET)) == list(range(14, 27))


async def test_off_sends_only_frames_sure_to_be_speech():
    orch = orchestrator(0, ONSET)
    assert await kept(orch, len(ONSET)) == list(range(22, 27))


async def test_a_short_gap_inside_speech_comes_back_once_and_in_order():
    """A breath between words is sent as it was, so the recogniser hears the gap, not words glued together."""
    probabilities = [0.9] * 3 + [0.1] * 3 + [0.9] * 3 + [0.1] * 4
    orch = orchestrator(256, probabilities)
    # trailing quiet after the last word is held back: no speech followed it
    assert await kept(orch, len(probabilities)) == list(range(9))


async def test_silence_without_speech_sends_nothing():
    orch = orchestrator(256, [0.05] * 30)
    assert await kept(orch, 30) == []


def test_the_agent_file_sets_it():
    assert Agent(prompt="hi", vad=VAD(pre_roll_ms=128)).config({}).vad.pre_roll_ms == 128
    assert Agent(prompt="hi").config({}).vad.pre_roll_ms == 256
    with pytest.raises(AgentError):
        VAD(pre_roll_ms=-5)
