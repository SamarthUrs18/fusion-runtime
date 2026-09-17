"""Reading the metadata header of a GGUF file without loading the model.

GGUF files describe themselves: architecture, context length and the chat
template are stored as key/value pairs before the weights. Reading them lets
the resolver pick a runtime and check a model before spending seconds (or
gigabytes) loading it.

Format: "GGUF" magic, uint32 version, uint64 tensor count, uint64 key/value
count, then each key (uint64 length + UTF-8 bytes), a uint32 value type and
the value. Little-endian. Large arrays (the tokenizer vocabulary) are skipped,
not read.
"""
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

MAGIC = b"GGUF"
MAX_ARRAY_ITEMS_KEPT = 64  # longer arrays (vocabularies, merges) are skipped
MAX_STRING_BYTES = 1 << 20  # a chat template is a few KB; anything huge means a corrupt file

_SCALARS = {  # value type -> struct format
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9


class GGUFError(ValueError):
    """Not a GGUF file, or a truncated / corrupt header."""


def read_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """All scalar and short-array metadata in the file header."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = _unpack(f, "<I")
        if version < 2:
            raise GGUFError(f"{path} uses GGUF version {version}; only version 2 and later are supported")
        _unpack(f, "<Q")  # tensor count
        count = _unpack(f, "<Q")
        metadata: Dict[str, Any] = {"GGUF.version": version}
        for _ in range(count):
            key = _read_string(f)
            value_type = _unpack(f, "<I")
            metadata[key] = _read_value(f, value_type)
        return {k: v for k, v in metadata.items() if v is not _SKIPPED}


class _Skipped:
    pass


_SKIPPED = _Skipped()


def _unpack(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file in GGUF header")
    return struct.unpack(fmt, data)[0]


def _read_string(f: BinaryIO) -> str:
    length = _unpack(f, "<Q")
    if length > MAX_STRING_BYTES:
        raise GGUFError(f"GGUF string of {length} bytes; the file is probably corrupt")
    data = f.read(length)
    if len(data) != length:
        raise GGUFError("unexpected end of file in GGUF header")
    return data.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, value_type: int):
    if value_type in _SCALARS:
        return _unpack(f, _SCALARS[value_type])
    if value_type == _STRING:
        return _read_string(f)
    if value_type == _ARRAY:
        item_type = _unpack(f, "<I")
        n = _unpack(f, "<Q")
        if n > MAX_ARRAY_ITEMS_KEPT:
            _skip_array(f, item_type, n)
            return _SKIPPED
        return [_read_value(f, item_type) for _ in range(n)]
    raise GGUFError(f"unknown GGUF value type {value_type}")


def _skip_array(f: BinaryIO, item_type: int, n: int) -> None:
    if item_type in _SCALARS:
        f.seek(struct.calcsize(_SCALARS[item_type]) * n, 1)
        return
    for _ in range(n):  # strings and nested arrays have variable size
        if item_type == _STRING:
            f.seek(_unpack(f, "<Q"), 1)
        else:
            _read_value(f, item_type)


def summarize(metadata: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """The fields the runtime and engine care about, under stable names."""
    architecture = metadata.get("general.architecture")

    def arch(key: str):
        return metadata.get(f"{architecture}.{key}") if architecture else None

    return {
        "architecture": architecture,
        "model_name": metadata.get("general.name"),
        "context_length": arch("context_length"),
        "chat_template": metadata.get("tokenizer.chat_template"),
        "split_count": metadata.get("split.count"),
        "kv_bytes_per_token": kv_bytes_per_token(
            arch("block_count"), arch("embedding_length"), arch("attention.head_count"),
            arch("attention.head_count_kv"),
        ),
    }


def kv_bytes_per_token(layers, embedding, heads, kv_heads, bytes_per_value: int = 2) -> Optional[int]:
    """Memory one token of context takes in the KV cache (fp16 by default).

    Keys and values, for every layer, for every KV head. Used to estimate how
    many conversations fit in memory (capacity and admission control).
    """
    if not all(isinstance(v, int) and v > 0 for v in (layers, embedding, heads)):
        return None
    kv_heads = kv_heads if isinstance(kv_heads, int) and kv_heads > 0 else heads
    return 2 * layers * kv_heads * (embedding // heads) * bytes_per_value
