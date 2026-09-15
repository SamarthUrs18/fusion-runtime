"""Model directory resolution: models must be found regardless of the
directory the server is started from."""
from pathlib import Path

import fusion_runtime.config as config
from fusion_runtime.config import (
    DEVELOPMENT_CONFIG,
    HYBRID_CONFIG,
    PRODUCTION_CONFIG,
    model_dir,
    resolve_model_path,
)


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path))
    assert model_dir() == tmp_path


def test_env_var_expands_home(monkeypatch):
    monkeypatch.setenv("FUSION_MODEL_DIR", "~/somewhere")
    assert model_dir() == Path.home() / "somewhere"


def test_falls_back_to_user_cache_without_source_models(monkeypatch, tmp_path):
    monkeypatch.delenv("FUSION_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # Pretend the package is installed somewhere with no models/ beside it
    fake_pkg = tmp_path / "site-packages" / "fusion_runtime" / "config.py"
    fake_pkg.parent.mkdir(parents=True)
    monkeypatch.setattr(config, "__file__", str(fake_pkg))
    assert model_dir() == tmp_path / "cache" / "fusion-runtime" / "models"


def test_relative_paths_resolve_under_model_dir_not_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path / "..")
    assert resolve_model_path("llm/x.gguf") == tmp_path / "models" / "llm" / "x.gguf"


def test_absolute_paths_are_kept(tmp_path):
    target = tmp_path / "custom.gguf"
    assert resolve_model_path(str(target)) == target


def test_profiles_use_model_dir_relative_paths():
    # Paths the downloader actually writes; no "models/" prefix (that was cwd-relative)
    for cfg in (DEVELOPMENT_CONFIG, PRODUCTION_CONFIG, HYBRID_CONFIG):
        assert not Path(cfg.tts.model).is_absolute()
        assert not cfg.tts.model.startswith("models/")
        assert cfg.tts.model == "tts/onnx/model.onnx"
    for cfg in (DEVELOPMENT_CONFIG, PRODUCTION_CONFIG):
        assert cfg.llm.model.startswith("llm/") and cfg.llm.model.endswith(".gguf")
