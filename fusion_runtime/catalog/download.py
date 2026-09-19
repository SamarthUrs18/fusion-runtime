"""Downloading models into the model directory: catalog entries, and any Hugging Face repo."""
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

HF_PREFIX = "hf:"
HF_TOKEN_ENV = "HF_TOKEN"  # for gated or private models; never stored in config
# Weights and the files needed to use them. Keeps big extras (other quantizations,
# training checkpoints, ONNX copies of a PyTorch model) off the disk.
HF_WANTED = ["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.gguf", "*.onnx", "*.onnx_data", "*.npz"]
HF_UNWANTED = ["*.msgpack", "*.h5", "*.tflite", "*.ot", "original/**", "*.pth"]
# A repo often holds the same model in several sizes (GGUF quantizations, ONNX
# variants). Files this big are one of those; smaller ones are the pieces every
# copy needs (configs, tokenizers, voice vectors) and always come along.
BIG_FILE_BYTES = 50_000_000

from fusion_runtime.catalog.entries import ModelEntry, format_size, is_installed, missing_files

# Keep this much free after a download so the machine doesn't end up at 0 bytes.
DISK_HEADROOM_BYTES = 500_000_000


class DownloadError(RuntimeError):
    pass


class ModelAccessDenied(DownloadError):
    """The repo is gated or private and the token is missing or not allowed."""


class ChooseAModel(DownloadError):
    """The repo holds the same model several times over; one has to be named."""


class NotEnoughDiskSpace(DownloadError):
    pass


def bytes_to_download(entry: ModelEntry, root: Path, force: bool = False) -> int:
    if entry.source == "torch_hub":
        return 0 if (is_installed(entry, root) and not force) else entry.size
    names = entry.files if force else missing_files(entry, root)
    return sum(entry.files[n] for n in names)


def check_disk_space(needed: int, root: Path) -> None:
    probe = root
    while not probe.exists():  # model dir may not exist yet
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed + DISK_HEADROOM_BYTES > free:
        raise NotEnoughDiskSpace(
            f"Need {format_size(needed)} (plus {format_size(DISK_HEADROOM_BYTES)} headroom) "
            f"but only {format_size(free)} is free at {probe}. "
            "Free up space or set FUSION_MODEL_DIR to a bigger disk."
        )


def pull(entry: ModelEntry, root: Path, force: bool = False, log: Callable[[str], None] = print) -> None:
    try:
        if entry.source == "huggingface":
            _pull_huggingface(entry, root, force, log)
        elif entry.source == "torch_hub":
            _pull_torch_hub(entry, force)
        else:
            raise DownloadError(f"{entry.id}: unknown source {entry.source!r} in the catalog")
    except DownloadError:
        raise
    except Exception as e:  # network, auth, missing file on the hub...
        raise DownloadError(f"{entry.id}: download failed: {type(e).__name__}: {e}") from e


def _pull_huggingface(entry: ModelEntry, root: Path, force: bool, log: Callable[[str], None]) -> None:
    from huggingface_hub import hf_hub_download

    target_dir = root / entry.local_dir
    names = list(entry.files) if force else missing_files(entry, root)
    for name in names:
        log(f"  {name} ({format_size(entry.files[name])})")
        hf_hub_download(
            repo_id=entry.repo,
            filename=name,
            revision=entry.revision,
            local_dir=target_dir,
            force_download=force,
        )
    still_wrong = missing_files(entry, root)
    if still_wrong:
        raise DownloadError(
            f"{entry.id}: {', '.join(still_wrong)} missing or the wrong size after download. "
            "Run the pull again with --force."
        )
    if entry.builds == "voices-v1.0.bin":
        _build_kokoro_voice_pack(target_dir, force)


def _build_kokoro_voice_pack(tts_dir: Path, force: bool) -> None:
    out = tts_dir / "voices-v1.0.bin"
    if out.is_file() and not force:
        return
    import numpy as np

    pack = {}
    for voice_file in sorted((tts_dir / "voices").glob("*.bin")):
        style = np.fromfile(voice_file, dtype=np.float32)
        if style.size == 510 * 256:  # one 256-dim style vector per token position
            pack[voice_file.stem] = style.reshape(510, 256)
    if not pack:
        raise DownloadError(f"No valid Kokoro voice files found in {tts_dir / 'voices'}")
    with open(out, "wb") as f:
        np.savez(f, **pack)  # open file object, so numpy doesn't append ".npz"


def hf_reference(ref: str) -> Optional[tuple]:
    """("owner/repo", revision or None, file or None) for an "hf:..." reference, else None.

        hf:owner/repo                          the whole repo
        hf:owner/repo@revision                 pinned to a commit
        hf:owner/repo/model-q4_k_m.gguf        one file in it
        hf:owner/repo@revision/onnx/model.onnx
    """
    if not ref.startswith(HF_PREFIX):
        return None
    rest, _, revision = ref[len(HF_PREFIX):].partition("@")
    if revision and "/" in revision:  # hf:owner/repo@rev/path/to/file
        revision, _, tail = revision.partition("/")
        rest = f"{rest}/{tail}"
    parts = [p for p in rest.split("/") if p]
    if len(parts) < 2:
        raise DownloadError(f"expected {HF_PREFIX}owner/repo, optionally @revision and a file, got {ref!r}")
    repo = "/".join(parts[:2])
    filename = "/".join(parts[2:]) or None
    return repo, revision or None, filename


def hf_files(repo: str, revision: Optional[str] = None, token: Optional[str] = None) -> list:
    """[(name, size)] of the files worth downloading from a repo."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo, revision=revision, files_metadata=True, token=token)
    return [(f.rfilename, f.size or 0) for f in (info.siblings or [])
            if any(Path(f.rfilename).match(p) for p in HF_WANTED)
            and not any(Path(f.rfilename).match(p) for p in HF_UNWANTED)]


def hf_local_dir(root: Path, repo: str, revision: Optional[str] = None) -> Path:
    """Where a downloaded Hugging Face repo lives, one folder per repo and revision."""
    name = re.sub(r"[^A-Za-z0-9._-]", "--", repo)
    return root / "hf" / (f"{name}@{revision}" if revision else name)


def is_hf_downloaded(root: Path, repo: str, revision: Optional[str] = None) -> bool:
    folder = hf_local_dir(root, repo, revision)
    return folder.is_dir() and any(p.is_file() for p in folder.rglob("*") if not p.name.startswith("."))


def pull_hf(ref: str, root: Path, log: Callable[[str], None] = print, force: bool = False) -> Path:
    """Download a Hugging Face model into the model directory, and return its folder.

    Repos often hold the same model several times over (GGUF quantizations, ONNX
    variants). Rather than fetching gigabytes of copies, name the one you want:
    `hf:owner/repo/model-q4_k_m.gguf`. Gated and private repos need a token in
    $HF_TOKEN.
    """
    repo, revision, filename = hf_reference(ref if ref.startswith(HF_PREFIX) else HF_PREFIX + ref)
    target = hf_local_dir(root, repo, revision)
    if is_hf_downloaded(root, repo, revision) and not force:
        return target

    from huggingface_hub import snapshot_download

    token = os.getenv(HF_TOKEN_ENV)
    try:
        wanted = _files_to_fetch(repo, revision, filename, token)
        check_disk_space(sum(size for _, size in wanted), root)
        log(f"  {repo}{'@' + revision if revision else ''} → {target}")
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=target,
            allow_patterns=[name for name, _ in wanted],
            token=token,
            force_download=force,
        )
        if (target / "voices").is_dir():  # Kokoro ships one file per voice; its runtime wants them packed
            _build_kokoro_voice_pack(target, force)
    except DownloadError:
        raise
    except Exception as e:
        name = type(e).__name__
        if name in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(e) or "403" in str(e):
            raise ModelAccessDenied(
                f"{repo} is gated or private. Accept its terms on huggingface.co/{repo}, then put an access "
                f"token in ${HF_TOKEN_ENV} (export {HF_TOKEN_ENV}=hf_...)"
            ) from e
        raise DownloadError(f"{repo}: download failed: {name}: {e}") from e
    return target


def _files_to_fetch(repo: str, revision: Optional[str], filename: Optional[str], token) -> list:
    """The named file plus the small files every copy needs, or the whole repo when it holds one model."""
    files = hf_files(repo, revision, token)
    if not files:
        raise DownloadError(f"{repo} has no model files we can use")
    small = [(name, size) for name, size in files if size < BIG_FILE_BYTES]
    big = [(name, size) for name, size in files if size >= BIG_FILE_BYTES]

    if filename:
        match = [(name, size) for name, size in files if name == filename]
        if not match:
            available = ", ".join(name for name, _ in big[:6]) or ", ".join(name for name, _ in files[:6])
            raise DownloadError(f"{repo} has no file {filename!r}. It has: {available}")
        return match + [f for f in small if f[0] != filename]
    if len(big) > 1:
        options = "\n".join(f"    hf:{repo}/{name}   ({size / 1e6:.0f} MB)" for name, size in sorted(big, key=lambda f: f[1]))
        raise ChooseAModel(
            f"{repo} holds {len(big)} versions of the model; downloading them all would be "
            f"{sum(size for _, size in big) / 1e9:.1f} GB. Name the one you want:\n{options}"
        )
    return files


def hf_expected_bytes(repo: str, revision: Optional[str] = None, token: Optional[str] = None,
                      filename: Optional[str] = None) -> Optional[int]:
    """Total size of what a download will fetch, or None when the hub can't be asked."""
    try:
        return sum(size for _, size in _files_to_fetch(repo, revision, filename, token)) or None
    except DownloadError:
        raise
    except Exception as e:
        if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            raise
        return None  # offline, or the hub changed: let the download itself report problems


def _pull_torch_hub(entry: ModelEntry, force: bool) -> None:
    import torch

    from fusion_runtime.catalog.entries import use_model_dir_for_torch_hub

    use_model_dir_for_torch_hub(entry.repo)
    torch.hub.load(entry.repo, "silero_vad", force_reload=force, trust_repo=True, verbose=False)
