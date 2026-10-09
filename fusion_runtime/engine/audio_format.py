"""Checking the caller's audio is what the pipeline expects: raw 16-bit mono PCM at the agreed rate.

Any binary WebSocket message used to be taken as PCM, so a mistake turned into noise without a
word: an odd byte count shifts every later sample by half; a WAV file arrives with a 44-byte header;
compressed audio (MP3, Ogg, WebM, FLAC) is read as loud static. Here:

    odd byte count   the stray byte is carried to the next chunk, so the stream stays aligned
    WAV header       stripped once if the format underneath is right; an error with the details if not
    compressed       an error naming the format, and what to send instead

A wrong sample rate in raw PCM can't be told from the bytes, so it isn't guessed at.
"""
import struct
from dataclasses import dataclass
from typing import Optional

# Leading bytes of formats people send by mistake. Only signatures of 3+ bytes: an untagged MP3 or
# AAC frame starts with just 2 (FF FB, FF F1), and raw PCM begins with those by chance about once
# in 13,000 calls, which would refuse a good call. Those files aren't caught here.
_COMPRESSED = (
    (b"ID3", "MP3"),
    (b"OggS", "Ogg (Opus or Vorbis)"),
    (b"\x1a\x45\xdf\xa3", "WebM"),
    (b"fLaC", "FLAC"),
)


class BadAudioFormat(ValueError):
    """The caller's audio can't be used as sent. The message says what arrived and what to send."""


@dataclass
class Intake:
    sample_rate: int = 16000
    _carry: bytes = b""
    _first: bool = True
    note: Optional[str] = None  # said once, then cleared: a fix fusion made for the client

    def expected(self) -> str:
        return f"raw 16-bit little-endian mono PCM at {self.sample_rate} Hz"

    def accept(self, data: bytes) -> bytes:
        """The PCM in this chunk, aligned to whole samples. Raises BadAudioFormat for audio it can't use."""
        if self._first and data:
            self._first = False
            data = self._check_first(data)
        data = self._carry + data
        if len(data) % 2:
            data, self._carry = data[:-1], data[-1:]
        else:
            self._carry = b""
        return data

    def _check_first(self, data: bytes) -> bytes:
        for magic, name in _COMPRESSED:
            if data.startswith(magic):
                raise BadAudioFormat(f"this is {name} audio; send {self.expected()} instead "
                                     "(decode it first, or capture raw PCM from the microphone)")
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return self._strip_wav(data)
        return data

    def _strip_wav(self, data: bytes) -> bytes:
        """A WAV file's header, if its audio is already what we expect; otherwise say what it is."""
        offset, fmt = 12, None
        while offset + 8 <= len(data):
            chunk_id, size = data[offset:offset + 4], struct.unpack("<I", data[offset + 4:offset + 8])[0]
            body = offset + 8
            if chunk_id == b"fmt " and size >= 16 and body + 16 <= len(data):
                fmt = struct.unpack("<HHIIHH", data[body:body + 16])
            elif chunk_id == b"data":
                if fmt is None:
                    break
                encoding, channels, rate, _, _, bits = fmt
                if encoding != 1 or channels != 1 or bits != 16 or rate != self.sample_rate:
                    kind = "PCM" if encoding == 1 else f"encoding {encoding}"
                    raise BadAudioFormat(
                        f"this is a WAV file with {kind}, {channels} channel(s), {bits}-bit, {rate} Hz; send "
                        f"{self.expected()} (convert it, e.g. ffmpeg -i in.wav -ac 1 -ar {self.sample_rate} "
                        "-f s16le out.raw)")
                self.note = "a WAV header was removed from the first audio message; send raw PCM without it"
                return data[body:]
            offset = body + size + (size % 2)
        raise BadAudioFormat(f"this looks like a WAV file but its header couldn't be read; send {self.expected()}")
