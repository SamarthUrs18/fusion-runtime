"""Engine-side streaming: rolling Whisper windows and sentence splitting, independent of any runtime."""
import asyncio

import pytest

from fusion_runtime.contract import InvalidRequest, Transcript
from fusion_runtime.engine.streaming import raise_if_error, rolling_transcripts
from fusion_runtime.engine.text import speakable_segments

ONE_SECOND = b"\x01\x00" * 16000


class RecordingTranscriber:
    def __init__(self):
        self.windows = []

    async def __call__(self, pcm: bytes) -> Transcript:
        self.windows.append(len(pcm) // 2)
        return Transcript(text=f"words {len(self.windows)}.", language="en", confidence=0.9)


async def chunks(*parts, before_each=None):
    for part in parts:
        if before_each:
            before_each()
        yield part


async def test_whole_turn_is_retranscribed_each_second_then_flushed():
    transcribe = RecordingTranscriber()
    half = ONE_SECOND[: len(ONE_SECOND) // 2]
    results = [r async for r in rolling_transcripts(chunks(half, half, half), transcribe)]
    assert transcribe.windows == [16000, 24000]  # cumulative window, then the leftover flushed as final
    assert [r.is_final for r in results] == [True, True]
    assert results[0].language == "en" and results[0].confidence == 0.9


async def test_reset_starts_a_new_turn_and_skips_a_stale_flush():
    transcribe = RecordingTranscriber()
    reset = asyncio.Event()
    stream = rolling_transcripts(chunks(ONE_SECOND, ONE_SECOND), transcribe, reset_signal=reset)
    await stream.__anext__()
    reset.set()  # the turn ended: the next chunk starts fresh
    await stream.__anext__()
    assert transcribe.windows == [16000, 16000]
    reset.set()  # a turn already consumed the tail: no spurious extra turn at the end
    assert [r async for r in stream] == []
    assert not reset.is_set()


async def test_window_is_capped():
    transcribe = RecordingTranscriber()
    async for _ in rolling_transcripts(chunks(*[ONE_SECOND] * 3), transcribe, max_window_s=2.0):
        pass
    assert max(transcribe.windows) == 32000


async def test_runtime_errors_in_result_slots_are_raised():
    with pytest.raises(InvalidRequest):
        raise_if_error(InvalidRequest("bad audio"))
    transcript = Transcript(text="ok")
    assert raise_if_error(transcript) is transcript


async def tokens(*parts):
    for part in parts:
        yield part


async def test_sentences_are_spoken_whole():
    segments = [s async for s in speakable_segments(tokens("We", " open", " at", " nine", ".", " See", " you", "!"))]
    assert segments == ["We open at nine.", " See you!"]


async def test_long_text_without_punctuation_is_split_and_the_rest_flushed():
    words = ["word "] * 30
    segments = [s async for s in speakable_segments(tokens(*words), max_chars=50)]
    assert all(len(s) >= 50 for s in segments[:-1]) and "".join(segments) == "".join(words)
