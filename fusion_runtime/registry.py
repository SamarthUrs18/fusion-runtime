"""Finding runtime classes by name.

A runtime is referenced in config as one of:

- a built-in name, e.g. "llama_cpp" (see BUILTIN_RUNTIMES)
- a registered name, from `register()` or an installed plugin
- an import path, "my_package.module:MyRuntime"

Plugins declare an entry point in their pyproject.toml, named "<stage>.<runtime>":

    [project.entry-points."fusion_runtime.runtimes"]
    "tts.my_engine" = "my_package.tts:MyEngineRuntime"

Classes are imported only when requested, so listing names stays cheap.
"""
import importlib
from importlib.metadata import entry_points
from typing import Dict, List, Tuple, Type

from fusion_runtime.contract.common import RUNTIME_STAGES, ModelRuntime, ModelSpec, Stage
from fusion_runtime.contract.llm import LLMRuntime
from fusion_runtime.contract.stt import STTRuntime
from fusion_runtime.contract.tts import TTSRuntime
from fusion_runtime.contract.turn import TurnDetector

ENTRY_POINT_GROUP = "fusion_runtime.runtimes"

STAGE_BASES: Dict[str, Type[ModelRuntime]] = {
    "stt": STTRuntime, "llm": LLMRuntime, "tts": TTSRuntime, "turn": TurnDetector,
}

# (stage, name) -> "module:Class"
BUILTIN_RUNTIMES: Dict[Tuple[str, str], str] = {
    ("stt", "ctranslate2"): "fusion_runtime.runtimes.ctranslate2.stt:CTranslate2STT",
    ("llm", "llama_cpp"): "fusion_runtime.runtimes.llama_cpp.llm:LlamaCppLLM",
    ("llm", "openai_http"): "fusion_runtime.runtimes.openai_http.llm:OpenAIHTTPLLM",
    ("tts", "onnx"): "fusion_runtime.runtimes.onnx.tts:OnnxTTS",
    ("turn", "silence"): "fusion_runtime.turns.silence:SilenceTurnDetector",
}

_registered: Dict[Tuple[str, str], str] = {}


class UnknownRuntime(LookupError):
    pass


def register(stage: Stage, name: str, target: str) -> None:
    """Register a runtime under a name at run time. `target` is "module:Class"."""
    _check_stage(stage)
    if ":" not in target:
        raise ValueError(f"target must look like 'module:Class', got {target!r}")
    _registered[(stage, name)] = target


def unregister(stage: Stage, name: str) -> None:
    _registered.pop((stage, name), None)


def _plugin_targets() -> Dict[Tuple[str, str], str]:
    found = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        stage, _, name = ep.name.partition(".")
        if stage in RUNTIME_STAGES and name:
            found[(stage, name)] = ep.value
    return found


def available_runtimes(stage: Stage) -> List[str]:
    _check_stage(stage)
    names = {name for (s, name) in {**BUILTIN_RUNTIMES, **_plugin_targets(), **_registered} if s == stage}
    return sorted(names)


def runtime_class(stage: Stage, runtime: str) -> Type[ModelRuntime]:
    """Resolve a built-in name, registered or plugin name, or 'module:Class' to a runtime class."""
    _check_stage(stage)
    if ":" in runtime:
        target = runtime
    else:
        target = (_registered.get((stage, runtime))
                  or _plugin_targets().get((stage, runtime))
                  or BUILTIN_RUNTIMES.get((stage, runtime)))
        if target is None:
            available = ", ".join(available_runtimes(stage)) or "none installed"
            raise UnknownRuntime(
                f"No {stage} runtime named {runtime!r}. Available: {available}. "
                "A custom runtime can be given as 'module:Class'."
            )
    cls = _import(target)
    base = STAGE_BASES[stage]
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise UnknownRuntime(f"{target} is not a {base.__name__} subclass")
    return cls


def create_runtime(spec: ModelSpec) -> ModelRuntime:
    """Construct (not load) the runtime a ModelSpec names."""
    return runtime_class(spec.stage, spec.runtime)(spec)


def _import(target: str):
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise UnknownRuntime(
            f"Can't import runtime module {module_name!r}: {e}. "
            "Install the package that provides it, or make sure it's on PYTHONPATH"
        ) from e
    try:
        obj = module
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError as e:
        raise UnknownRuntime(f"{module_name!r} has no attribute {attr!r}") from e
    return obj


def _check_stage(stage: str) -> None:
    if stage not in RUNTIME_STAGES:
        raise ValueError(f"stage must be one of {', '.join(RUNTIME_STAGES)}, got {stage!r}")
