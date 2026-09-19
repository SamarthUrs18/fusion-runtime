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
    runtime: str = ""  # engine that runs it: llama_cpp | ctranslate2 | onnx | torch_hub
    family: Optional[str] = None  # processing the file format doesn't describe (whisper, kokoro)
    languages: Optional[tuple] = None  # None = many, or read from the model itself
    voices: tuple = ()

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
    catalog = {}
    for model_id, fields in raw.items():
        for key in ("languages", "voices"):  # tuples keep entries hashable and read-only
            if key in fields:
                fields[key] = tuple(fields[key])
        catalog[model_id] = ModelEntry(id=model_id, **fields)
    return catalog


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
    local_providers = {"stt": Provider.FASTER_WHISPER, "llm": Provider.LLAMA_CPP, "tts": Provider.KOKORO}
    wanted = {}
    for stage, provider in local_providers.items():
        stage_config = getattr(config, stage)
        ref = stage_config.model
        if ref.startswith(("http://", "https://")) or (not stage_config.runtime and stage_config.provider != provider):
            continue  # served by an endpoint, or by something the catalog doesn't cover
        wanted[stage] = {ref, f"stt/{ref}"} if stage == "stt" else {ref}
    needed = []
    for entry in catalog.values():
        if entry.stage == "vad":
            if config.vad.provider == Provider.SILERO:
                needed.append(entry)
        elif entry.stage in wanted and (entry.path in wanted[entry.stage] or entry.id in wanted[entry.stage]):
            needed.append(entry)
    return sorted(needed, key=lambda e: STAGES.index(e.stage))


def torch_hub_dir() -> Path:
    """Where the voice detector's checkout lives.

    Inside the model directory, not PyTorch's own cache in the home directory,
    so that one directory holds everything a server needs. On a deployment that
    means one volume: mount it, and a fresh container downloads nothing. With
    the detector somewhere else, a pod that looked fully provisioned would still
    reach out to GitHub on every cold start — a network call at exactly the
    moment there is nothing to fall back on.

    TORCH_HOME still wins, for anyone who already points PyTorch somewhere.
    """
    if os.getenv("TORCH_HOME"):
        return Path(os.environ["TORCH_HOME"]).expanduser() / "hub"
    from fusion_runtime.config import model_dir

    return model_dir() / "torch-hub"


def legacy_torch_hub_dir() -> Path:
    """Where PyTorch keeps its own cache, and where this used to live."""
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache / "torch" / "hub"


def use_model_dir_for_torch_hub(repo: Optional[str] = None) -> Path:
    """Point torch.hub at that directory. Called before any hub load, so the
    place we download to and the place we check are never different.

    With `repo`, a copy already sitting in PyTorch's cache is adopted rather
    than fetched again. That matters beyond saving a few megabytes: a machine
    that can reach the rest of the internet but not GitHub's download host would
    otherwise lose a detector it already had.
    """
    import torch

    path = torch_hub_dir()
    torch.hub.set_dir(str(path))
    if repo:
        _adopt_existing_checkout(repo, path)
    return path


def _adopt_existing_checkout(repo: str, path: Path) -> None:
    import shutil

    legacy = legacy_torch_hub_dir()
    if legacy == path:
        return
    owner, _, name = repo.partition("/")
    checkout = f"{owner}_{name}_master"
    source, target = legacy / checkout, path / checkout
    if target.exists() or not source.is_dir():
        return
    path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)  # a couple of megabytes, once
    trusted = legacy / "trusted_list"
    if trusted.is_file() and not (path / "trusted_list").exists():
        shutil.copy2(trusted, path / "trusted_list")


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
