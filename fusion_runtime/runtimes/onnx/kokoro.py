"""Family spec for Kokoro: how to turn text into Kokoro's ONNX inputs and back.

An ONNX file is only a graph; it doesn't say how to prepare its inputs. This
spec holds exactly that knowledge for the Kokoro family (every Kokoro v1.0
export and voice pack), so the ONNX runtime itself stays model-agnostic.

Uses kokoro-onnx for the phonemizer (bundled espeak data, no system install),
the ONNX session and the voice pack, and runs inference directly for control
over the style vector.

Files: the graph (e.g. tts/onnx/model.onnx) and the voice pack
voices-v1.0.bin next to it or one folder up (`frun models pull` builds it),
or the `voices_path` option.
"""
import importlib.metadata
import os
from pathlib import Path
from typing import List, Optional, Tuple

SAMPLE_RATE = 24000
STYLE_DIM = 256

# Kokoro voice ids start with a language letter: af_heart = American English, female.
VOICE_LANGUAGES = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br",
}


# Short codes to the phonemizer's names
ESPEAK_ALIASES = {"en": "en-us", "fr": "fr-fr", "pt": "pt-br"}


class KokoroFamily:
    sample_rate = SAMPLE_RATE

    def __init__(self, model_path: Path, options):
        self.model_path = model_path
        self.options = options
        self._kokoro = None
        self.degraded: Optional[str] = None
        self.provider_warnings: List[Tuple[str, str]] = []

    def load(self) -> None:
        """Blocking: call from a worker thread."""
        from kokoro_onnx import Kokoro

        from fusion_runtime.contract import ModelNotFound

        if not self.model_path.is_file():
            raise ModelNotFound(f"Kokoro model not found at {self.model_path}. Run: frun models pull")
        candidates = []
        if self.options.get("voices_path"):
            candidates.append(Path(self.options["voices_path"]).expanduser())
        candidates += [self.model_path.parent / "voices-v1.0.bin", self.model_path.parent.parent / "voices-v1.0.bin"]
        voices = next((p for p in candidates if p.is_file()), None)
        if voices is None:
            raise ModelNotFound(
                f"Kokoro voice pack (voices-v1.0.bin) not found in: {', '.join(str(c) for c in candidates)}. "
                "Run: frun models pull"
            )
        session = onnx_session(self.model_path, found=self.provider_warnings)
        self._kokoro = Kokoro.from_session(session, str(voices))
        # Running, but slower than it should be: health() reports it for as long as the server runs
        self.degraded = next((w for w, _ in self.provider_warnings if "CPU" in w), None)

    @property
    def voices(self) -> Tuple[str, ...]:
        return tuple(sorted(self._kokoro.voices)) if self._kokoro is not None else ()

    @property
    def languages(self) -> Tuple[str, ...]:
        found = {VOICE_LANGUAGES[v[0]] for v in self.voices if v[:1] in VOICE_LANGUAGES}
        return tuple(sorted(found)) or ("en-us",)

    def language_for(self, voice: str, language: Optional[str]) -> str:
        """The phonemizer's language code: from the request if given ("en" means American), else the voice's."""
        if language:
            code = language.lower().replace("_", "-")
            return ESPEAK_ALIASES.get(code, code)
        return VOICE_LANGUAGES.get(voice[:1], "en-us")

    def synthesize(self, text: str, voice: str, speed: float, language: Optional[str]) -> bytes:
        """Blocking: text to 16-bit PCM at 24 kHz. Call from a worker thread."""
        import numpy as np

        k = self._kokoro
        phonemes = k.tokenizer.phonemize(text, self.language_for(voice, language))
        tokens = k.tokenizer.tokenize(phonemes)
        if not tokens:  # whitespace or punctuation only: a tiny silence
            return np.zeros(240, dtype=np.int16).tobytes()
        pack = k.voices[voice]
        # The style vector depends on input length: row (tokens - 1) of the voice pack
        style = pack[min(len(tokens), len(pack)) - 1].reshape(1, STYLE_DIM).astype(np.float32)
        outputs = k.sess.run(None, {
            "input_ids": np.array([[0, *tokens, 0]], dtype=np.int64),
            "style": style,
            "speed": np.array([speed], dtype=np.float32),
        })
        audio = outputs[0].ravel()
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


GPU_BUILD = "onnxruntime-gpu"


def onnx_session(model_path: Path, found: Optional[list] = None):
    """The model's onnxruntime session: CUDA when this onnxruntime has it, else the CPU.

    kokoro-onnx asks for every provider the GPU build lists, TensorRT first, and
    TensorRT then fails to load on most machines with a page of errors before
    falling back. CUDA is what's wanted, so ask for it, and say plainly when it
    didn't load rather than let text-to-speech run on the CPU unnoticed (seconds
    a sentence instead of milliseconds). ONNX_PROVIDER still picks one by name.
    """
    import onnxruntime as rt

    available = rt.get_available_providers()
    named = os.getenv("ONNX_PROVIDER")
    if named:
        providers = [named] + (["CPUExecutionProvider"] if named != "CPUExecutionProvider" else [])
    elif "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    options = rt.SessionOptions()
    options.log_severity_level = 3  # errors only; a provider that didn't load is reported below, once
    session = rt.InferenceSession(str(model_path), sess_options=options, providers=providers)
    warnings = provider_warnings(providers, session.get_providers(), _installed_builds(),
                                 cpu_on_purpose=named == "CPUExecutionProvider")
    if found is not None:
        found.extend(warnings)
    for warning in warnings:
        from fusion_runtime.telemetry import telemetry

        telemetry.emit("tts.on_cpu" if "CPU" in warning[0] else "tts.onnxruntime", level="warning",
                       stage="tts", hint=warning[0], fix=warning[1])
    return session


def provider_warnings(asked: List[str], got: List[str], builds: List[str],
                      cpu_on_purpose: bool = False) -> List[Tuple[str, str]]:
    """(what's wrong, the fix) for an onnxruntime that can't use the GPU it was installed for.

    ONNX_PROVIDER=CPUExecutionProvider on a GPU machine is a choice (leaving the GPU to the LLM),
    not a broken install, so it isn't reported as one.
    """
    warnings = []
    if "onnxruntime" in builds and GPU_BUILD in builds:
        # Both install the same `onnxruntime` module, so whichever went in last wins, usually the
        # CPU one, pulled in by another package (pod test, 25 Sep).
        warnings.append((
            "both onnxruntime and onnxruntime-gpu are installed; they share one module, so the GPU one "
            "may be overwritten",
            "pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu"))
    wanted_gpu = [p for p in asked if p != "CPUExecutionProvider"]
    if wanted_gpu and not any(p in got for p in wanted_gpu):
        warnings.append((
            f"text-to-speech is running on the CPU: {wanted_gpu[0]} didn't load",
            "onnxruntime-gpu 1.30+ needs CUDA 13 while torch uses CUDA 12; see "
            "https://fusion-runtime.dev/docs#gpu"))
    elif GPU_BUILD in builds and not wanted_gpu and not cpu_on_purpose:
        warnings.append((
            "text-to-speech is running on the CPU: onnxruntime-gpu is installed but this onnxruntime "
            "has no CUDA provider",
            "pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu"))
    return warnings


def _installed_builds() -> List[str]:
    found = []
    for name in ("onnxruntime", GPU_BUILD):
        try:
            importlib.metadata.distribution(name)
            found.append(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return found
