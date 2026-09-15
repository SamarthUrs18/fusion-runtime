"""Downloading catalog models into the model directory."""
import shutil
from pathlib import Path
from typing import Callable

from fusion_runtime.catalog.entries import ModelEntry, format_size, is_installed, missing_files

# Keep this much free after a download so the machine doesn't end up at 0 bytes.
DISK_HEADROOM_BYTES = 500_000_000


class DownloadError(RuntimeError):
    pass


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


def _pull_torch_hub(entry: ModelEntry, force: bool) -> None:
    import torch

    torch.hub.load(entry.repo, "silero_vad", force_reload=force, trust_repo=True, verbose=False)
