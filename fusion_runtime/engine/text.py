"""Splitting a streamed LLM reply into pieces worth speaking.

TTS gets whole sentences, never word fragments: fragments sound robotic and
waste synthesis. A piece is sent when a token ends a sentence or line, or
when the text grows past `max_chars` without one (long run-on replies).
The rest is flushed when the reply ends.
"""
from typing import AsyncIterator

SENTENCE_ENDINGS = (".", "!", "?", "\n")


async def speakable_segments(tokens: AsyncIterator[str], max_chars: int = 100) -> AsyncIterator[str]:
    buffer = ""
    async for token in tokens:
        buffer += token
        if len(buffer) >= max_chars or token.endswith(SENTENCE_ENDINGS):
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
