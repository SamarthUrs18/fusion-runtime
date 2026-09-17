"""Text handling that works across languages: splitting replies for speech, and words.

Punctuation here only shapes what the agent says (whole sentences to TTS). It
never decides whether the caller has finished speaking: that is turn detection.

No per-language code paths: the punctuation tables cover Latin, Devanagari
(। ॥), CJK (。！？), Arabic and Urdu (؟ ۔), Armenian (։), Ethiopic (።) and
others, and a mark that can't appear in a language simply never matches.
"""
import re
import unicodedata
from typing import AsyncIterator, List

# Marks that end a sentence without doubt, wherever they appear
HARD_SENTENCE_ENDS = ("。", "！", "？", "।", "॥", "؟", "۔", "։", "።", "፧", "\n")
# Marks that end a sentence only when followed by a space or the end of text:
# "3.5", "e.g." and "U.S.A" must not be split mid-way
SOFT_SENTENCE_ENDS = (".", "!", "?", "…")
# Where a long run-on sentence can be broken without cutting a word
CLAUSE_BREAKS = (",", ";", ":", "，", "、", "；", "：", "،", "؛")
CLOSING_MARKS = "\"'”’)]}»」』"
# Sent through a token stream when a reply is complete, so its last sentence is spoken right away
END_OF_REPLY = ""
# Sent when the caller interrupted the reply: its unspoken words are thrown away, so they
# can't be spoken at the start of the next reply
REPLY_CUT_OFF = "\x00reply-cut-off\x00"

# Scripts written without spaces between words: each character counts as a word
_NO_SPACE_SCRIPTS = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u0e00-\u0e7f\u0e80-\u0eff\u1000-\u109f\u1780-\u17ff]"
)


async def speakable_segments(tokens: AsyncIterator[str], max_chars: int = 100) -> AsyncIterator[str]:
    """Split a streamed reply into whole sentences for TTS.

    Hard sentence ends split right away. A soft end (".", "!", "?") splits
    once the next token shows a space or line break after it, so decimals and
    abbreviations stay whole. That costs one token of waiting (about 10 ms)
    and only at a sentence end. Past `max_chars` without a sentence end, the
    text is split at the last clause break (comma and friends, in any script)
    or, if there is none, as it is. Whatever is left is flushed at the end.
    """
    buffer = ""
    async for token in tokens:
        if token == END_OF_REPLY:
            if buffer.strip():
                yield buffer
            buffer = ""
            continue
        if token == REPLY_CUT_OFF:
            buffer = ""
            continue
        if buffer.rstrip(CLOSING_MARKS).endswith(SOFT_SENTENCE_ENDS) and token[:1].isspace():
            yield buffer
            buffer = token.lstrip(" \t")
            if not buffer:
                continue
        else:
            buffer += token
        stripped = buffer.rstrip(CLOSING_MARKS)
        if stripped.endswith(HARD_SENTENCE_ENDS):
            yield buffer
            buffer = ""
        elif len(buffer) >= max_chars and not stripped.endswith(SOFT_SENTENCE_ENDS):
            cut = max(buffer.rfind(mark) for mark in CLAUSE_BREAKS)
            if cut > 0:
                yield buffer[: cut + 1]
                buffer = buffer[cut + 1:]
            else:
                yield buffer
                buffer = ""
    if buffer.strip():
        yield buffer


def words(text: str) -> List[str]:
    """Lowercased words with punctuation removed, in any script.

    Punctuation and symbols become spaces (by Unicode category, so "।" and "，"
    count too). Combining marks stay, so Devanagari or Thai words aren't
    broken apart. In scripts written without spaces (Chinese, Japanese, Thai),
    each character counts as a word, so phrases can still be compared word by word.
    """
    cleaned = "".join(
        " " if unicodedata.category(ch)[0] in "PSZC" and ch not in "‌‍" else ch
        for ch in text.lower()
    )
    cleaned = _NO_SPACE_SCRIPTS.sub(lambda m: f" {m.group(0)} ", cleaned)
    return cleaned.split()
