"""G.711 μ-law: how telephone audio travels, 8 bits a sample at 8 kHz.

Phone providers (Twilio, Plivo, Exotel) stream calls in it. Python's audioop did this, and was
removed in 3.13, so it's here: the standard bias-and-segment encoding, vectorised with numpy.
"""
import numpy as np

_BIAS = 0x84


def _decode_table() -> np.ndarray:
    codes = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(codes & 0x80, -magnitude, magnitude).astype(np.int16)


_DECODE = _decode_table()


def decode(data: bytes) -> np.ndarray:
    """μ-law bytes to int16 samples."""
    return _DECODE[np.frombuffer(data, dtype=np.uint8)]


_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])


def encode(samples: np.ndarray) -> bytes:
    """int16 samples (or anything castable) to μ-law bytes, matching the reference (audioop) exactly.

    Works on 14-bit values the way G.711 is specified: shift, take the magnitude, add the bias,
    find the segment, keep four bits of mantissa, invert.
    """
    x = np.asarray(samples, dtype=np.int32) >> 2
    negative = x < 0
    magnitude = np.minimum(np.where(negative, -x, x), 8159) + (_BIAS >> 2)
    segment = np.searchsorted(_SEGMENT_ENDS, magnitude, side="left")
    code = np.where(segment >= 8, 0x7F, (segment << 4) | ((magnitude >> (segment + 1)) & 0x0F))
    return (code ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()


# A phone line's silence: what encode() gives for 0, and what providers pad with
SILENCE = encode(np.zeros(1))
