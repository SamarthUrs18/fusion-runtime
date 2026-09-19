"""Model catalog: pinned entries, profile mapping, installed checks and downloads.
No network: Hugging Face and torch hub calls are replaced with fakes."""
import numpy as np
import pytest

import huggingface_hub
from fusion_runtime.catalog import (
    ModelEntry,
    UnknownModelError,
    entries_for_profile,
    format_size,
    get_entries,
    is_installed,
    load_catalog,
)
from fusion_runtime.catalog import download
from fusion_runtime.catalog.entries import STAGES
from fusion_runtime.config import DEVELOPMENT_CONFIG, HYBRID_CONFIG, PRODUCTION_CONFIG


def test_catalog_entries_are_pinned_and_complete():
    catalog = load_catalog()
    assert catalog
    for entry in catalog.values():
        assert entry.stage in STAGES
        assert entry.total_bytes > 0
        if entry.source == "huggingface":
            assert entry.revision and len(entry.revision) == 40, f"{entry.id} must pin a commit"
            assert entry.files and all(size > 0 for size in entry.files.values())
            assert entry.path.startswith(entry.local_dir)


def test_every_profile_model_exists_in_catalog():
    # Guards against config pointing at a file nothing downloads (the old 7B bug)
    assert [e.id for e in entries_for_profile(DEVELOPMENT_CONFIG)] == [
        "whisper-tiny.en", "qwen2.5-0.5b-q4", "kokoro-v1.0", "silero-vad"]
    assert [e.id for e in entries_for_profile(PRODUCTION_CONFIG)] == [
        "whisper-tiny.en", "qwen2.5-7b-q4", "kokoro-v1.0", "silero-vad"]


def test_hybrid_profile_downloads_no_llm():
    assert "llm" not in {e.stage for e in entries_for_profile(HYBRID_CONFIG)}


def test_unknown_ids_list_what_is_available():
    with pytest.raises(UnknownModelError, match="Available: whisper-tiny.en"):
        get_entries(["nope"])


def test_format_size():
    assert format_size(2317) == "2 KB"
    assert format_size(75_537_502) == "76 MB"
    assert format_size(4_683_073_632) == "4.7 GB"


def _entry(**overrides) -> ModelEntry:
    fields = dict(id="tiny", stage="llm", description="test", license="MIT", source="huggingface",
                  repo="org/tiny", revision="0" * 40, local_dir="llm", path="llm/a.gguf",
                  files={"a.gguf": 10, "sub/b.bin": 20})
    fields.update(overrides)
    return ModelEntry(**fields)


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


def test_installed_requires_every_file_at_the_right_size(tmp_path):
    entry = _entry()
    assert not is_installed(entry, tmp_path)
    _write(tmp_path / "llm" / "a.gguf", 10)
    _write(tmp_path / "llm" / "sub" / "b.bin", 19)  # truncated download
    assert not is_installed(entry, tmp_path)
    _write(tmp_path / "llm" / "sub" / "b.bin", 20)
    assert is_installed(entry, tmp_path)


def test_installed_requires_built_file(tmp_path):
    entry = _entry(builds="voices-v1.0.bin", files={"a.gguf": 10})
    _write(tmp_path / "llm" / "a.gguf", 10)
    assert not is_installed(entry, tmp_path)
    _write(tmp_path / "llm" / "voices-v1.0.bin", 1)
    assert is_installed(entry, tmp_path)


def test_torch_hub_entry_checks_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    entry = load_catalog()["silero-vad"]
    assert not is_installed(entry, tmp_path)
    _write(tmp_path / "hub" / "snakers4_silero-vad_master" / "hubconf.py", 1)
    assert is_installed(entry, tmp_path)


def _fake_hub(calls, sizes=None):
    def fake_hf_hub_download(repo_id, filename, revision, local_dir, force_download):
        calls.append((repo_id, filename, revision))
        _write(local_dir / filename, (sizes or {}).get(filename, 10 if filename == "a.gguf" else 20))
    return fake_hf_hub_download


def test_pull_fetches_only_missing_files_at_pinned_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub(calls))
    entry = _entry()
    _write(tmp_path / "llm" / "a.gguf", 10)  # already there
    download.pull(entry, tmp_path, log=lambda _: None)
    assert calls == [("org/tiny", "sub/b.bin", "0" * 40)]
    assert is_installed(entry, tmp_path)


def test_pull_rejects_wrong_size_after_download(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub([], sizes={"a.gguf": 3}))
    with pytest.raises(download.DownloadError, match="wrong size"):
        download.pull(_entry(), tmp_path, log=lambda _: None)


def test_network_errors_become_download_errors(tmp_path, monkeypatch):
    def offline(**_):
        raise ConnectionError("no route to host")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(download.DownloadError, match="tiny: download failed: ConnectionError"):
        download.pull(_entry(), tmp_path, log=lambda _: None)


def test_kokoro_voice_pack_is_built_from_voice_files(tmp_path, monkeypatch):
    voice_bytes = 510 * 256 * 4

    def fake(repo_id, filename, revision, local_dir, force_download):
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".bin"):
            np.full(510 * 256, 0.5, dtype=np.float32).tofile(target)
        else:
            _write(target, 10)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    entry = _entry(stage="tts", local_dir="tts", path="tts/onnx/model.onnx", builds="voices-v1.0.bin",
                   files={"onnx/model.onnx": 10, "voices/af_heart.bin": voice_bytes, "voices/am_michael.bin": voice_bytes})
    download.pull(entry, tmp_path, log=lambda _: None)
    pack = np.load(tmp_path / "tts" / "voices-v1.0.bin")
    assert sorted(pack.files) == ["af_heart", "am_michael"]
    assert pack["af_heart"].shape == (510, 256)
    assert is_installed(entry, tmp_path)


def test_disk_space_check(tmp_path, monkeypatch):
    class Usage:
        free = 1_000_000_000
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: Usage)
    download.check_disk_space(100_000_000, tmp_path / "not" / "created" / "yet")
    with pytest.raises(download.NotEnoughDiskSpace, match="FUSION_MODEL_DIR"):
        download.check_disk_space(4_700_000_000, tmp_path)


# ---- where the voice detector lives ----------------------------------------------

def test_the_detector_sits_with_the_other_models(tmp_path, monkeypatch):
    """One directory holds everything a server needs, so a deployment is one volume."""
    from fusion_runtime.catalog.entries import torch_hub_dir

    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path / "models"))
    assert torch_hub_dir() == tmp_path / "models" / "torch-hub"


def test_torch_home_still_wins(tmp_path, monkeypatch):
    from fusion_runtime.catalog.entries import torch_hub_dir

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "elsewhere"))
    assert torch_hub_dir() == tmp_path / "elsewhere" / "hub"


def test_a_checkout_already_on_the_machine_is_adopted_not_refetched(tmp_path, monkeypatch):
    """A machine that can reach the internet but not GitHub's download host would
    otherwise lose a detector it already had."""
    from fusion_runtime.catalog import entries

    legacy = tmp_path / "cache" / "torch" / "hub"
    (legacy / "snakers4_silero-vad_master").mkdir(parents=True)
    (legacy / "snakers4_silero-vad_master" / "hubconf.py").write_text("# the checkout\n")
    (legacy / "trusted_list").write_text("snakers4\n")
    monkeypatch.setattr(entries, "legacy_torch_hub_dir", lambda: legacy)

    destination = tmp_path / "models" / "torch-hub"
    entries._adopt_existing_checkout("snakers4/silero-vad", destination)

    assert (destination / "snakers4_silero-vad_master" / "hubconf.py").is_file()
    assert (destination / "trusted_list").is_file()


def test_adoption_never_overwrites_what_is_there(tmp_path, monkeypatch):
    from fusion_runtime.catalog import entries

    legacy = tmp_path / "cache" / "torch" / "hub"
    (legacy / "snakers4_silero-vad_master").mkdir(parents=True)
    (legacy / "snakers4_silero-vad_master" / "hubconf.py").write_text("# old\n")
    monkeypatch.setattr(entries, "legacy_torch_hub_dir", lambda: legacy)

    destination = tmp_path / "models" / "torch-hub"
    (destination / "snakers4_silero-vad_master").mkdir(parents=True)
    (destination / "snakers4_silero-vad_master" / "hubconf.py").write_text("# current\n")
    entries._adopt_existing_checkout("snakers4/silero-vad", destination)

    assert (destination / "snakers4_silero-vad_master" / "hubconf.py").read_text() == "# current\n"
