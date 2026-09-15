"""Reading the model catalog and checking what's installed.

Kept free of heavy imports (no torch, no huggingface_hub, no pydantic) so
`frun models list` stays fast.
"""
import os
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files as package_files
from pathlib import Path
from typing import Optional

STAGES = ("stt", "llm", "tts", "vad")


@dataclass(frozen=True)
class ModelEntry:
    id: str
    stage: str
    description: str
    license: str
    source: str  # "huggingface" | "torch_hub"
    repo: str
    revision: Optional[str] = None
    local_dir: str = ""
    path: str = ""
    files: dict = field(default_factory=dict)  # repo file name -> size in bytes
    builds: Optional[str] = None  # extra file assembled after download, under local_dir
    size: int = 0  # for sources without a file list

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values()) if self.files else self.size


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.1f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.0f} MB"
    return f"{max(num_bytes / 1e3, 1):.0f} KB"


class UnknownModelError(ValueError):
    pass


def load_catalog() -> dict[str, ModelEntry]:
    raw = tomllib.loads(package_files("fusion_runtime.catalog").joinpath("models.toml").read_text())
    return {model_id: ModelEntry(id=model_id, **fields) for model_id, fields in raw.items()}


def get_entries(ids: list[str], catalog: Optional[dict[str, ModelEntry]] = None) -> list[ModelEntry]:
    catalog = catalog or load_catalog()
    unknown = [i for i in ids if i not in catalog]
    if unknown:
        raise UnknownModelError(
            f"Unknown model{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}. "
            f"Available: {', '.join(catalog)}"
        )
    return [catalog[i] for i in ids]


def entries_for_profile(config, catalog: Optional[dict[str, ModelEntry]] = None) -> list[ModelEntry]:
    """The catalog entries a PipelineConfig needs downloaded, in stage order."""
    from fusion_runtime.config import Provider

    catalog = catalog or load_catalog()
    wanted = {
        "stt": f"stt/{config.stt.model}" if config.stt.provider == Provider.FASTER_WHISPER else None,
        "llm": config.llm.model if config.llm.provider == Provider.LLAMA_CPP else None,
        "tts": config.tts.model if config.tts.provider == Provider.KOKORO else None,
    }
    needed = []
    for entry in catalog.values():
        if entry.stage == "vad":
            if config.vad.provider == Provider.SILERO:
                needed.append(entry)
        elif wanted.get(entry.stage) and entry.path == wanted[entry.stage]:
            needed.append(entry)
    return sorted(needed, key=lambda e: STAGES.index(e.stage))


def torch_hub_dir() -> Path:
    """Same location torch.hub.get_dir() uses, without importing torch."""
    if os.getenv("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]).expanduser() / "hub"
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache / "torch" / "hub"


def missing_files(entry: ModelEntry, root: Path) -> list[str]:
    """Repo files not yet on disk with the expected size."""
    base = root / entry.local_dir
    return [
        name for name, size in entry.files.items()
        if not (base / name).is_file() or (base / name).stat().st_size != size
    ]


def is_installed(entry: ModelEntry, root: Path) -> bool:
    if entry.source == "torch_hub":
        owner, name = entry.repo.split("/")
        return (torch_hub_dir() / f"{owner}_{name}_master" / "hubconf.py").is_file()
    if missing_files(entry, root):
        return False
    return entry.builds is None or (root / entry.local_dir / entry.builds).is_file()
