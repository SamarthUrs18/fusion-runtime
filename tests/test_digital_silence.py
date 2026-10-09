"""Exact-zero audio never reaches speech-to-text.

Asif Ali's vendor benchmark (fusion-runtime 0.1.1 post, Oct 2026): one recogniser wrote fluent
sentences from digital zeros on 15 of 15 probes, and nothing from real room tone (0 of 4). The VAD
usually keeps zeros away from Whisper; this covers the case where it failed to load.
"""
import random
import struct

from fusion_runtime.config import PipelineConfig
from fusion_runtime.contract import Transcript
from fusion_runtime.engine import PipelineOrchestrator
from fusion_runtime.telemetry import SessionTrace

ZEROS = b"\x00" * 640  # 20 ms of a muted mic at 16 kHz


def room_tone(samples=320, level=12, seed=1):
    """A quiet room: tiny, never-quite-zero noise."""
    rng = random.Random(seed)
    return struct.pack(f"<{samples}h", *(rng.randint(-level, level) or 1 for _ in range(samples)))


async def chunks(*parts):
    for part in parts:
        yield part


async def test_zeros_are_dropped_and_quiet_rooms_are_not():
    trace = SessionTrace()
    trace.listening_turn()
    tone = room_tone()
    out = [c async for c in PipelineOrchestrator._drop_digital_silence(chunks(ZEROS, tone, ZEROS, tone), trace)]
    assert out == [tone, tone]
    assert trace.listening.counts["stt_zero_chunks_dropped"] == 2


class RecordingSTT:
    def __init__(self):
        self.heard = []

    async def transcribe(self, requests):
        self.heard += [r.audio for r in requests]
        return [Transcript(text="", duration_s=len(r.audio) / 32000) for r in requests]


async def test_with_the_voice_detector_missing_zeros_still_never_reach_the_recogniser():
    """The real failure: the VAD didn't load, so every chunk was passed straight through."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = PipelineConfig()
    orch.stt = RecordingSTT()

    async def no_vad():
        return None

    orch._load_vad_frame_model = no_vad
    audio = [ZEROS] * 100 + [room_tone(seed=s) for s in range(20)] + [ZEROS] * 100
    async for _ in orch._stt_stage(chunks(*audio)):
        pass
    assert orch.stt.heard, "the room tone should still have been transcribed"
    assert not any(ZEROS in heard for heard in orch.stt.heard)
