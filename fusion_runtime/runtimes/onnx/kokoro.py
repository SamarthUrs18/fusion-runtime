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
from pathlib import Path
from typing import Optional, Tuple

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
        self._kokoro = Kokoro(str(self.model_path), str(voices))

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
