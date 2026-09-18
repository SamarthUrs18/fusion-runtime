"""An agent in one Python file: what it says, which models it uses, how it takes turns.

    # agent.py
    from fusion_runtime import Agent, LLM, TTS, Turns

    agent = Agent(
        prompt="You take orders for ShopKart. Keep answers to one short sentence.",
        llm=LLM("qwen2.5-0.5b-q4", max_tokens=200),
        tts=TTS("kokoro-v1.0", voice="af_heart"),
        turns=Turns(wait_ms=500),
    )

    frun up agent.py

The agent is the product, so it lives in code: versioned, reviewed, and able to
hold the tools it will gain later. Secrets never belong here — those come from
environment variables (`api_key_env`, `HF_TOKEN`).

A stage takes a model name, with settings when it needs them:

    stt="whisper-tiny.en"                       just the model
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=200)  model plus settings

A model is named the way the resolver understands, optionally with the runtime
in front:

    "qwen2.5-0.5b-q4"                          a catalog model (`frun models list`)
    "./models/my-finetune.gguf"                a file
    "hf:Systran/faster-whisper-small"          a Hugging Face model
    "vllm:hf:mistralai/Mistral-7B-Instruct"    that model, served by vLLM
    "http://localhost:8000/v1"                 an OpenAI-compatible endpoint

Settings the runtime's config already knows (`max_tokens`, `voice`, `n_ctx`,
...) are applied to it; anything else is handed to that runtime untouched, so a
new engine flag needs no change here.
"""
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

DEFAULT_PROMPT = "You are a helpful voice assistant. Answer briefly."

# Runtimes that can appear in front of a model reference ("vllm:hf:org/repo").
# "module:Class" works too, for plugins.
KNOWN_RUNTIMES = ("llama_cpp", "ctranslate2", "onnx", "openai_http", "vllm", "llama_server")


class AgentError(ValueError):
    """The agent file can't be used as written."""


@dataclass(frozen=True)
class Stage:
    """One stage's model and its settings. Use STT, LLM or TTS, not this."""

    ref: str
    runtime: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    stage: str = ""

    def __init__(self, ref: str, runtime: Optional[str] = None, **options: Any) -> None:
        name = type(self).__name__
        if not isinstance(ref, str) or not ref.strip():
            raise AgentError(f"{name} needs a model, for example {name}('kokoro-v1.0')")
        object.__setattr__(self, "ref", ref.strip())
        object.__setattr__(self, "runtime", runtime.strip() if isinstance(runtime, str) else None)
        object.__setattr__(self, "options", dict(options))
        object.__setattr__(self, "stage", getattr(type(self), "STAGE", ""))

    @classmethod
    def of(cls, value: Union[str, "Stage", None], stage: str) -> Optional["Stage"]:
        """Accept a plain model name, or the stage class for this stage."""
        if value is None:
            return None
        if isinstance(value, str):
            return _STAGE_CLASSES[stage](value)
        if isinstance(value, Stage):
            if value.stage and value.stage != stage:
                raise AgentError(f"{type(value).__name__}(...) was given as the {stage} model; "
                                 f"use {_STAGE_CLASSES[stage].__name__}(...) instead")
            return value
        raise AgentError(f"{stage} takes a model name or {_STAGE_CLASSES[stage].__name__}(...), "
                         f"got {type(value).__name__}")

    def split(self) -> tuple:
        """(runtime, model reference).

            "vllm:hf:org/repo"        → ("vllm", "hf:org/repo")         a runtime in front
            "my_pkg.turns:Detector"   → ("my_pkg.turns:Detector", "")   a plugin runtime
            "hf:org/repo", a URL, an id, a path → (None, as written)    the format decides
        """
        if self.runtime:
            return self.runtime, self.ref
        prefix, _, rest = self.ref.partition(":")
        if rest and prefix in KNOWN_RUNTIMES:
            return prefix, rest
        if rest and "." in prefix and "/" not in rest and ":" not in rest:
            return self.ref, ""  # "module:Class": the whole thing names the runtime
        return None, self.ref


class STT(Stage):
    """The speech recognition model: STT("whisper-tiny.en", beam_size=5)."""

    STAGE = "stt"


class LLM(Stage):
    """The language model: LLM("qwen2.5-0.5b-q4", max_tokens=200)."""

    STAGE = "llm"


class TTS(Stage):
    """The voice: TTS("kokoro-v1.0", voice="af_heart", speed=1.1)."""

    STAGE = "tts"


_STAGE_CLASSES = {"stt": STT, "llm": LLM, "tts": TTS}


@dataclass(frozen=True)
class Turns:
    """Turn taking: when the agent answers, and when it stops for the caller.

        Turns(wait_ms=500, interrupt_after_ms=300)
        Turns("my_package.turns:MyDetector", wait_ms=400)   # with a turn detector model

    A detector predicts whether the caller is done, which shortens or lengthens
    the wait (see fusion_runtime.contract.turn). Without one, the agent answers
    after `wait_ms` of silence.
    """

    detector: Optional[Stage] = None
    wait_ms: Optional[int] = None  # silence before the agent answers
    interrupt_after_ms: Optional[int] = None  # speech over the agent before it stops
    resume_window_ms: Optional[int] = None  # speaking again this soon continues the same turn
    options: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, detector: Union[str, Stage, None] = None, *, wait_ms: Optional[int] = None,
                 interrupt_after_ms: Optional[int] = None, resume_window_ms: Optional[int] = None,
                 **options: Any) -> None:
        if isinstance(detector, str):
            detector = Stage(detector)
        elif detector is not None and not isinstance(detector, Stage):
            raise AgentError(f"Turns takes a detector name or Stage(...), got {type(detector).__name__}")
        object.__setattr__(self, "detector", detector)
        for name, value in (("wait_ms", wait_ms), ("interrupt_after_ms", interrupt_after_ms),
                            ("resume_window_ms", resume_window_ms)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise AgentError(f"Turns({name}=...) must be a whole number of milliseconds, got {value!r}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "options", dict(options))


@dataclass(frozen=True)
class VAD:
    """Which audio counts as speech: VAD(threshold=0.6).

    Feeds both turn taking and interruption. Silero is the only detector today,
    so a model reference isn't accepted yet.
    """

    threshold: Optional[float] = None  # higher ignores more background noise
    min_speech_ms: Optional[int] = None
    min_silence_ms: Optional[int] = None

    def __init__(self, model: Optional[str] = None, *, threshold: Optional[float] = None,
                 min_speech_ms: Optional[int] = None, min_silence_ms: Optional[int] = None) -> None:
        if model is not None:
            raise AgentError(
                "VAD models can't be swapped yet (Silero is built in); VAD(threshold=..., "
                "min_speech_ms=..., min_silence_ms=...) sets its behaviour"
            )
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise AgentError(f"VAD(threshold=...) is between 0 and 1, got {threshold!r}")
        object.__setattr__(self, "threshold", threshold)
        for name, value in (("min_speech_ms", min_speech_ms), ("min_silence_ms", min_silence_ms)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise AgentError(f"VAD({name}=...) must be a whole number of milliseconds, got {value!r}")
            object.__setattr__(self, name, value)


@dataclass
class Agent:
    """Everything that makes one voice agent."""

    prompt: str = DEFAULT_PROMPT
    name: Optional[str] = None
    stt: Union[str, Stage, None] = None
    llm: Union[str, Stage, None] = None
    tts: Union[str, Stage, None] = None
    language: Optional[str] = None  # what callers speak; None keeps the profile's setting
    turns: Optional[Turns] = None
    vad: Optional[VAD] = None
    tools: Sequence[Any] = ()
    profile: str = "development"  # the defaults everything above is applied to
    source: Optional[Path] = None  # the file it was loaded from, when it came from one

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise AgentError("prompt must not be empty: it's what the agent is told to do")
        for stage in ("stt", "llm", "tts"):
            setattr(self, stage, Stage.of(getattr(self, stage), stage))
        if self.turns is not None and not isinstance(self.turns, Turns):
            raise AgentError(f"turns takes Turns(...), got {type(self.turns).__name__}")
        if self.vad is not None and not isinstance(self.vad, VAD):
            raise AgentError(f"vad takes VAD(...), got {type(self.vad).__name__}")
        if self.tools:
            raise AgentError(
                "tool calling isn't supported yet, so tools= can't be used. "
                "Everything else in the agent works; tools are the next piece being built"
            )

    def config(self, environ: Optional[Mapping[str, str]] = None):
        """This agent as a PipelineConfig: profile defaults, the agent on top, then the environment."""
        from fusion_runtime.config import PROFILES, with_env_overrides

        if self.profile not in PROFILES:
            raise AgentError(f"unknown profile {self.profile!r}; choose one of: {', '.join(PROFILES)}")
        config = PROFILES[self.profile]
        parts: Dict[str, Any] = {}

        for stage in ("stt", "llm", "tts"):
            model = getattr(self, stage)
            stage_config = getattr(config, stage)
            updates: Dict[str, Any] = {}
            if model is not None:
                runtime, ref = model.split()
                fields, options = _split_settings(model.options, stage_config)
                updates.update(model=ref, runtime=runtime, options=options, **fields)
            if self.language is not None and stage in ("stt", "tts"):
                updates["language"] = self.language
            if updates:
                parts[stage] = stage_config.model_copy(update=updates)

        if self.turns is not None:
            turn_config = config.turn_detection
            updates = {}
            for setting, field_name in (("wait_ms", "min_silence_ms"),
                                        ("interrupt_after_ms", "barge_in_min_speech_ms"),
                                        ("resume_window_ms", "resume_window_ms")):
                value = getattr(self.turns, setting)
                if value is not None:
                    updates[field_name] = value
            if self.turns.detector is not None:
                runtime, ref = self.turns.detector.split()
                updates.update(runtime=runtime or ref, model=ref if runtime else None)
            fields, options = _split_settings(self.turns.options, turn_config)
            updates.update(fields)
            if options:
                updates["options"] = options
            parts["turn_detection"] = turn_config.model_copy(update=updates)

        if self.vad is not None:
            updates = {k: v for k, v in {
                "threshold": self.vad.threshold,
                "min_speech_ms": self.vad.min_speech_ms,
                "min_silence_ms": self.vad.min_silence_ms,
            }.items() if v is not None}
            if updates:
                parts["vad"] = config.vad.model_copy(update=updates)

        # Environment last: a deployment can point the LLM elsewhere or change the
        # waits without editing the agent (see fusion_runtime.config.with_env_overrides).
        return with_env_overrides(config.model_copy(update=parts) if parts else config, environ)

    def describe(self) -> Dict[str, Any]:
        """A short summary for logs and `frun up` output. Never includes the prompt text."""
        def model_of(stage: str) -> Optional[str]:
            model = getattr(self, stage)
            return model.ref if model is not None else None

        detector = self.turns.detector if self.turns is not None else None
        return {k: v for k, v in {
            "agent": self.name or (self.source.stem if self.source else None),
            "source": str(self.source) if self.source else None,
            "prompt_chars": len(self.prompt),
            "stt": model_of("stt"), "llm": model_of("llm"), "tts": model_of("tts"),
            "turn_detector": detector.ref if detector is not None else None,
            "language": self.language,
            "profile": self.profile,
        }.items() if v is not None}


def _split_settings(options: Mapping[str, Any], stage_config) -> tuple:
    """Settings the config declares become config fields; the rest go to the runtime."""
    known = set(type(stage_config).model_fields) - {"model", "runtime", "family", "options", "provider"}
    fields = {k: v for k, v in options.items() if k in known}
    return fields, {k: v for k, v in options.items() if k not in known}


def load_agent(path: Union[str, Path]) -> Agent:
    """Run an agent file and return the Agent it defines.

    The file is ordinary Python: it may import anything and define helpers. It
    needs one Agent, either assigned to `agent` or as the only one defined.
    """
    file_path = Path(path).expanduser().resolve()
    if file_path.is_dir():
        raise AgentError(f"{file_path} is a folder; point at one agent file (running a folder of agents comes later)")
    if not file_path.is_file():
        raise AgentError(f"no agent file at {file_path}")
    if file_path.suffix != ".py":
        raise AgentError(f"an agent file must be a .py file, got {file_path.name}")

    module_name = f"fusion_agent_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise AgentError(f"can't load {file_path} as Python")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # The file's own folder first, so `import my_tools` next to the agent works
    added_path = str(file_path.parent)
    inserted = added_path not in sys.path
    if inserted:
        sys.path.insert(0, added_path)
    try:
        spec.loader.exec_module(module)
    except AgentError:
        raise
    except Exception as e:
        raise AgentError(f"{file_path.name} failed while loading: {type(e).__name__}: {e}") from e
    finally:
        if inserted:
            sys.path.remove(added_path)
        sys.modules.pop(module_name, None)

    agent = getattr(module, "agent", None)
    if not isinstance(agent, Agent):
        found = [value for value in vars(module).values() if isinstance(value, Agent)]
        if len(found) == 1:
            agent = found[0]
        elif not found:
            raise AgentError(
                f"{file_path.name} defines no Agent. Add one:\n"
                "    from fusion_runtime import Agent\n"
                '    agent = Agent(prompt="...")'
            )
        else:
            raise AgentError(f"{file_path.name} defines {len(found)} agents; name the one to run `agent`")
    agent.source = file_path
    return agent
