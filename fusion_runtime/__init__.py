"""
fusion-runtime: Low-latency voice AI inference runtime.
"""
import importlib
from importlib.metadata import PackageNotFoundError, version as _package_version
from typing import TYPE_CHECKING

try:
    __version__ = _package_version("fusion-runtime")  # the one real version lives in pyproject.toml
except PackageNotFoundError:  # source checkout that was never pip-installed
    __version__ = "0.0.0+unknown"

# Public names load on first use, so light entry points like `frun --help`
# don't pull in pydantic, numpy and the whole pipeline just by importing
# the package. `from fusion_runtime import PipelineOrchestrator` still works.
_EXPORTS = {
    "PipelineConfig": "fusion_runtime.config",
    "STTConfig": "fusion_runtime.config",
    "LLMConfig": "fusion_runtime.config",
    "TTSConfig": "fusion_runtime.config",
    "VADConfig": "fusion_runtime.config",
    "TurnDetectionConfig": "fusion_runtime.config",
    "Provider": "fusion_runtime.config",
    "DEVELOPMENT_CONFIG": "fusion_runtime.config",
    "PRODUCTION_CONFIG": "fusion_runtime.config",
    "HYBRID_CONFIG": "fusion_runtime.config",
    "Agent": "fusion_runtime.agent",
    "STT": "fusion_runtime.agent",
    "LLM": "fusion_runtime.agent",
    "TTS": "fusion_runtime.agent",
    "Turns": "fusion_runtime.agent",
    "VAD": "fusion_runtime.agent",
    "load_agent": "fusion_runtime.agent",
    "PipelineOrchestrator": "fusion_runtime.engine",
    "run_single_turn": "fusion_runtime.engine",
}
__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # let editors and type checkers see the real names
    from fusion_runtime.config import (
        DEVELOPMENT_CONFIG,
        HYBRID_CONFIG,
        PRODUCTION_CONFIG,
        LLMConfig,
        PipelineConfig,
        Provider,
        STTConfig,
        TTSConfig,
        TurnDetectionConfig,
        VADConfig,
    )
    from fusion_runtime.agent import LLM, STT, TTS, Agent, Turns, VAD, load_agent
    from fusion_runtime.engine import PipelineOrchestrator, run_single_turn


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'fusion_runtime' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache so the lookup happens once
    return value
