"""Turning a model reference into a runtime, a family and a concrete model.

Users name a model; they never name code. The resolver works out which
runtime runs it and how, from data: the catalog, the file format and the
file's own metadata.

A reference is one of:

- a catalog id:            "qwen2.5-0.5b-q4", "whisper-tiny.en", "kokoro-v1.0"
- a local path:            "./my-model.gguf", "~/models/whisper-hindi", "llm/x.gguf" (relative to the model directory)
- a Hugging Face repo:     "hf:owner/repo" or "hf:owner/repo@revision" (must already be downloaded)
- an HTTP endpoint:        "https://api.groq.com/openai/v1", "http://localhost:8000/v1"

Detection by format, never by model name:

    .gguf file                    → llama_cpp   (architecture, context length, chat template read from the file)
    CTranslate2 folder            → ctranslate2 (Whisper family: language support read from the vocabulary)
    .onnx file / folder           → onnx        (needs a family: from the catalog, a hint, or recognizable files)
    http(s):// URL                → openai_http
    safetensors folder            → not run in process: serve it with vLLM and point at its URL

An explicit `runtime` (a built-in name, a plugin name or "module:Class")
skips detection; a reference that isn't a local file is then handed to that
runtime unchanged, so plugins can take any kind of model reference.

Resolution reads headers only and does no network access. It can take tens
of milliseconds for a large GGUF vocabulary, so call it off the event loop.
"""
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from fusion_runtime.catalog import gguf
from fusion_runtime.catalog.entries import ModelEntry, is_installed, load_catalog, missing_files
from fusion_runtime.contract.common import (
    STAGES, InvalidRequest, ModelNotFound, ModelSpec, Stage, UnsupportedModel,
)

# Which stages each built-in runtime serves.
RUNTIME_STAGES: Dict[str, Tuple[str, ...]] = {
    "llama_cpp": ("llm",),
    "ctranslate2": ("stt",),
    "onnx": ("tts",),
    "openai_http": ("llm",),  # STT/TTS over HTTP when a customer needs it
}

_SPLIT_PART = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$")
_WHISPER_ENGLISH_ONLY_VOCAB = 51864  # multilingual Whisper vocabularies are 51865 (v1/v2) or 51866 (v3)


@dataclass(frozen=True)
class ResolvedModel:
    """Everything known about a model before loading it."""

    spec: ModelSpec
    source: str  # "catalog" | "path" | "huggingface" | "url" | "plugin"
    format: Optional[str]  # "gguf" | "ctranslate2" | "onnx" | "http" | None (plugin-defined)
    catalog_id: Optional[str] = None
    languages: Optional[Tuple[str, ...]] = None  # None = many / unknown
    voices: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)  # facts read from the model (architecture, context length, ...)

    def describe(self) -> Dict[str, Any]:
        """For logs, telemetry and `frun doctor`: no file contents, no secrets."""
        out: Dict[str, Any] = {
            "stage": self.spec.stage, "runtime": self.spec.runtime, "format": self.format,
            "family": self.spec.family, "source": self.source, "catalog_id": self.catalog_id,
            "model": self.spec.model, "languages": list(self.languages) if self.languages else None,
            "voices": len(self.voices) or None,
        }
        for key, value in self.metadata.items():
            if key == "chat_template":
                out["chat_template"] = bool(value)
            elif value is not None:
                out[key] = value
        return {k: v for k, v in out.items() if v is not None}


def resolve(
    stage: Stage,
    ref: str,
    *,
    runtime: Optional[str] = None,
    family: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
    catalog: Optional[Dict[str, ModelEntry]] = None,
    root: Optional[Path] = None,
) -> ResolvedModel:
    """Resolve `ref` for `stage`. Raises ModelNotFound, UnsupportedModel or InvalidRequest."""
    if stage not in STAGES:
        raise InvalidRequest(f"stage must be one of {', '.join(STAGES)}, got {stage!r}")
    ref = (ref or "").strip()
    if not ref:
        raise InvalidRequest(f"no {stage} model given")
    options = dict(options or {})
    if root is None:
        from fusion_runtime.config import model_dir
        root = model_dir()
    catalog = load_catalog() if catalog is None else catalog

    if ref.startswith(("http://", "https://")):
        chosen = runtime or "openai_http"
        _check_runtime_stage(chosen, stage, ref)
        return ResolvedModel(ModelSpec(stage, chosen, ref, family, options), "url", "http")

    if ref.startswith("hf:"):
        return _resolve_hf(stage, ref, runtime, family, options, catalog, root)

    entry = catalog.get(ref)
    if entry is None:
        entry = next((e for e in catalog.values() if e.path and e.path == ref), None)
    if entry is not None and entry.stage == stage:
        return _from_catalog(entry, root, runtime, family, options)
    if entry is not None:
        raise InvalidRequest(f"{ref!r} is a {entry.stage} model, but it was given as the {stage} model")

    path = _local_path(ref, root)
    if path is not None:
        return _from_path(stage, path, runtime, family, options, source="path")

    if runtime is not None and runtime not in RUNTIME_STAGES:
        # A plugin or custom runtime: it knows what its model references mean.
        return ResolvedModel(ModelSpec(stage, runtime, ref, family, options), "plugin", None)

    raise ModelNotFound(_not_found_message(stage, ref, root, catalog))


def resolve_stage_config(stage: Stage, stage_config, *, catalog=None, root: Optional[Path] = None) -> ResolvedModel:
    """Resolve one stage of a PipelineConfig to a runtime.

    A stage names its runtime directly (`runtime=`, with any model reference),
    or through `provider` (the older form, mapped to a runtime here). Settings
    become runtime options; API keys never do (only the name of the
    environment variable holding one).
    """
    from fusion_runtime.config import Provider

    provider = stage_config.provider
    settings = stage_config.model_dump(
        exclude={"provider", "model", "api_key", "api_base", "runtime", "family", "options"})
    settings = {k: v for k, v in settings.items() if v is not None}
    settings.update(stage_config.options)
    family = stage_config.family
    ref = stage_config.model

    if stage_config.runtime:
        return resolve(stage, ref, runtime=stage_config.runtime, family=family, options=settings,
                       catalog=catalog, root=root)
    if stage == "llm" and (provider == Provider.OPENAI or ref.startswith(("http://", "https://"))):
        if ref.startswith(("http://", "https://")):
            url = ref
        else:
            settings["model_name"] = ref
            url = getattr(stage_config, "api_base", None) or "https://api.openai.com/v1"
        return resolve("llm", url, runtime="openai_http", options=settings, catalog=catalog, root=root)
    if stage == "stt" and provider == Provider.FASTER_WHISPER:
        if "/" not in ref and not ref.startswith(("~", ".")) and ref not in (catalog or load_catalog()):
            ref = f"stt/{ref}"  # faster-whisper size names live under stt/ in the model directory
        return resolve("stt", ref, runtime="ctranslate2", family=family or "whisper", options=settings,
                       catalog=catalog, root=root)
    if stage == "llm" and provider == Provider.LLAMA_CPP:
        return resolve("llm", ref, runtime="llama_cpp", options=settings, catalog=catalog, root=root)
    if stage == "tts" and provider == Provider.KOKORO:
        return resolve("tts", ref, runtime="onnx", family=family or "kokoro", options=settings,
                       catalog=catalog, root=root)
    raise UnsupportedModel(f"no runtime for provider {getattr(provider, 'value', provider)!r} on the {stage} stage")


# ---- sources ------------------------------------------------------------------------

def _from_catalog(entry: ModelEntry, root: Path, runtime, family, options) -> ResolvedModel:
    if entry.source != "huggingface":
        raise UnsupportedModel(f"{entry.id} is loaded by the engine directly, not through a runtime")
    if not is_installed(entry, root):
        missing = missing_files(entry, root)
        detail = f" (missing: {', '.join(missing[:3])}{'…' if len(missing) > 3 else ''})" if missing else ""
        raise ModelNotFound(f"{entry.id} isn't downloaded yet{detail}. Run: frun models pull {entry.id}")
    resolved = _from_path(entry.stage, root / entry.path, runtime or entry.runtime or None,
                          family or entry.family, options, source="catalog")
    return ResolvedModel(
        resolved.spec, "catalog", resolved.format, catalog_id=entry.id,
        languages=entry.languages or resolved.languages, voices=entry.voices or resolved.voices,
        metadata=resolved.metadata,
    )


def _resolve_hf(stage, ref, runtime, family, options, catalog, root) -> ResolvedModel:
    repo, _, revision = ref[3:].partition("@")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise InvalidRequest(f"expected hf:owner/repo or hf:owner/repo@revision, got {ref!r}")
    for entry in catalog.values():
        if entry.repo == repo and (not revision or entry.revision == revision) and entry.stage == stage:
            return _from_catalog(entry, root, runtime, family, options)
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as e:  # pragma: no cover - installed with faster-whisper
        raise ModelNotFound(f"{ref}: huggingface_hub isn't installed") from e
    try:
        local = snapshot_download(repo, revision=revision or None, local_files_only=True)
    except LocalEntryNotFoundError:
        raise ModelNotFound(
            f"{repo} isn't in the local Hugging Face cache. Download it first "
            f"(huggingface-cli download {repo}); downloading any repo from frun comes later"
        ) from None
    resolved = _from_path(stage, Path(local), runtime, family, options, source="huggingface")
    return resolved


def _local_path(ref: str, root: Path) -> Optional[Path]:
    candidate = Path(ref).expanduser()
    for path in ((candidate,) if candidate.is_absolute() else (candidate, root / candidate)):
        if path.exists():
            return path.resolve()
    return None


# ---- formats --------------------------------------------------------------------------

def _from_path(stage, path: Path, runtime, family, options, source: str) -> ResolvedModel:
    if not path.exists():
        raise ModelNotFound(f"{path} doesn't exist")

    if runtime is not None and runtime not in RUNTIME_STAGES:
        return ResolvedModel(ModelSpec(stage, runtime, str(path), family, options), "plugin", None)

    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            first_parts = [p for p in ggufs if not _SPLIT_PART.match(p.name) or _SPLIT_PART.match(p.name)["part"] == "00001"]
            if len(first_parts) > 1:
                raise InvalidRequest(
                    f"{path} holds {len(first_parts)} GGUF models ({', '.join(p.name for p in first_parts[:3])}); "
                    "point at one file"
                )
            return _gguf(stage, first_parts[0], runtime, family, options, source)
        if (path / "model.bin").is_file() and (path / "config.json").is_file():
            return _ctranslate2(stage, path, runtime, family, options, source)
        if any(path.glob("*.safetensors")):
            raise UnsupportedModel(
                f"{path} is a safetensors model, which needs its architecture's code to run. "
                "Serve it with vLLM (or another OpenAI-compatible server) and use its URL as the model"
            )
        onnx_files = sorted(path.glob("*.onnx")) or sorted(path.glob("onnx/*.onnx"))
        if onnx_files:
            return _onnx(stage, onnx_files[0], runtime, family, options, source)
        raise UnsupportedModel(f"can't tell what kind of model {path} holds (no .gguf, CTranslate2 or .onnx files)")

    suffix = path.suffix.lower()
    if suffix == ".gguf":
        return _gguf(stage, path, runtime, family, options, source)
    if suffix == ".onnx":
        return _onnx(stage, path, runtime, family, options, source)
    if suffix == ".safetensors":
        return _from_path(stage, path.parent, runtime, family, options, source)
    raise UnsupportedModel(f"unrecognized model file {path.name}; supported: .gguf, .onnx, CTranslate2 folders")


def _gguf(stage, path: Path, runtime, family, options, source) -> ResolvedModel:
    chosen = runtime or "llama_cpp"
    _check_runtime_stage(chosen, stage, str(path))
    split = _SPLIT_PART.match(path.name)
    if split and split["part"] != "00001":
        first = path.with_name(f"{split['stem']}-00001-of-{split['total']}.gguf")
        raise InvalidRequest(f"{path.name} is part {int(split['part'])} of a split model; use the first part: {first.name}")
    try:
        facts = gguf.summarize(gguf.read_metadata(path))
    except (gguf.GGUFError, OSError) as e:
        raise UnsupportedModel(f"can't read {path.name}: {e}") from e
    if split:
        total = int(split["total"])
        missing = [f"{split['stem']}-{i:05d}-of-{split['total']}.gguf" for i in range(2, total + 1)
                   if not path.with_name(f"{split['stem']}-{i:05d}-of-{split['total']}.gguf").is_file()]
        if missing:
            raise ModelNotFound(f"{path.name} is split into {total} files; missing: {', '.join(missing)}")
    return ResolvedModel(ModelSpec(stage, chosen, str(path), family, options), source, "gguf", metadata=facts)


def _ctranslate2(stage, path: Path, runtime, family, options, source) -> ResolvedModel:
    chosen = runtime or "ctranslate2"
    _check_runtime_stage(chosen, stage, str(path))
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError) as e:
        raise UnsupportedModel(f"can't read {path / 'config.json'}: {e}") from e
    is_whisper = "lang_ids" in config or "suppress_ids_begin" in config
    family = family or ("whisper" if is_whisper else None)
    if family != "whisper":
        raise UnsupportedModel(f"{path} is a CTranslate2 model but not a Whisper model; only Whisper is supported for STT")
    languages = None
    vocabulary = path / "vocabulary.txt"
    if vocabulary.is_file():
        with open(vocabulary, "rb") as f:
            if sum(1 for _ in f) <= _WHISPER_ENGLISH_ONLY_VOCAB:
                languages = ("en",)
    return ResolvedModel(ModelSpec(stage, chosen, str(path), family, options), source, "ctranslate2",
                         languages=languages)


def _onnx(stage, path: Path, runtime, family, options, source) -> ResolvedModel:
    chosen = runtime or "onnx"
    _check_runtime_stage(chosen, stage, str(path))
    model_root = path.parent.parent if path.parent.name == "onnx" else path.parent
    voices = tuple(sorted(p.stem for p in (model_root / "voices").glob("*.bin")))
    if family is None and voices and stage == "tts":
        family = "kokoro"  # Kokoro ships a voices/ folder of style vectors next to the graph
    if family is None:
        raise UnsupportedModel(
            f"{path.name} is an ONNX graph, which doesn't say how to prepare its inputs. "
            "Set the model family (for example family=\"kokoro\")"
        )
    return ResolvedModel(ModelSpec(stage, chosen, str(path), family, options), source, "onnx", voices=voices)


def _check_runtime_stage(runtime: str, stage: str, ref: str) -> None:
    stages = RUNTIME_STAGES.get(runtime)
    if stages is not None and stage not in stages:
        raise UnsupportedModel(f"{ref} runs on {runtime}, which serves {'/'.join(stages)}, not {stage}")


def _not_found_message(stage: str, ref: str, root: Path, catalog: Dict[str, ModelEntry]) -> str:
    ids = [e.id for e in catalog.values() if e.stage == stage]
    close = difflib.get_close_matches(ref, ids, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else (f" Known {stage} models: {', '.join(ids)}." if ids else "")
    return (f"No {stage} model {ref!r}: not a catalog id, a file (checked here and under {root}), "
            f"an hf:owner/repo or a URL.{hint}")
