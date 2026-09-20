"""Runtime registry: names, 'module:Class' paths and plugins resolve to runtime classes."""
from types import SimpleNamespace

import pytest
from fusion_runtime import registry
from fusion_runtime.registry import (
    UnknownRuntime,
    available_runtimes,
    create_runtime,
    runtime_class,
)
from fusion_runtime.testing.fakes import FakeLLMRuntime, FakeTTSRuntime, fake_spec

FAKE_TTS = "fusion_runtime.testing.fakes:FakeTTSRuntime"


def test_module_class_path_resolves():
    assert runtime_class("tts", FAKE_TTS) is FakeTTSRuntime


def test_registered_name_resolves_and_can_be_removed():
    registry.register("tts", "fake", FAKE_TTS)
    try:
        assert runtime_class("tts", "fake") is FakeTTSRuntime
        assert "fake" in available_runtimes("tts")
    finally:
        registry.unregister("tts", "fake")
    assert "fake" not in available_runtimes("tts")


def test_unknown_name_lists_what_is_available():
    registry.register("tts", "fake", FAKE_TTS)
    try:
        with pytest.raises(UnknownRuntime, match=r"No tts runtime named 'nope'\. Available: .*fake"):
            runtime_class("tts", "nope")
    finally:
        registry.unregister("tts", "fake")


def test_class_for_the_wrong_stage_is_rejected():
    with pytest.raises(UnknownRuntime, match="not a LLMRuntime"):
        runtime_class("llm", FAKE_TTS)


def test_non_runtime_target_is_rejected():
    with pytest.raises(UnknownRuntime, match="not a TTSRuntime"):
        runtime_class("tts", "fusion_runtime.config:model_dir")


def test_missing_module_or_attribute():
    with pytest.raises(UnknownRuntime, match="Can't import runtime module"):
        runtime_class("tts", "no_such_package.x:Y")
    with pytest.raises(UnknownRuntime, match="has no attribute"):
        runtime_class("tts", "fusion_runtime.testing.fakes:Missing")


def test_register_requires_module_class_target():
    with pytest.raises(ValueError, match="module:Class"):
        registry.register("tts", "bad", "just_a_name")


def test_bad_stage():
    with pytest.raises(ValueError, match="stage must be one of"):
        runtime_class("vision", FAKE_TTS)


def test_plugins_from_entry_points(monkeypatch):
    plugins = [
        SimpleNamespace(name="llm.fake_plugin", value="fusion_runtime.testing.fakes:FakeLLMRuntime"),
        SimpleNamespace(name="not-a-stage-name", value="x:y"),  # ignored
    ]
    monkeypatch.setattr(registry, "entry_points", lambda group: plugins)
    assert "fake_plugin" in available_runtimes("llm")
    assert runtime_class("llm", "fake_plugin") is FakeLLMRuntime


def test_registered_name_wins_over_plugin(monkeypatch):
    monkeypatch.setattr(registry, "entry_points",
                        lambda group: [SimpleNamespace(name="tts.fake", value="x.y:Missing")])
    registry.register("tts", "fake", FAKE_TTS)
    try:
        assert runtime_class("tts", "fake") is FakeTTSRuntime
    finally:
        registry.unregister("tts", "fake")


def test_create_runtime_from_spec():
    spec = fake_spec("tts")
    spec = type(spec)(stage="tts", runtime=FAKE_TTS, model="fake")
    runtime = create_runtime(spec)
    assert isinstance(runtime, FakeTTSRuntime) and runtime.spec is spec
