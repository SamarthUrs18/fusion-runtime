"""
fusion-runtime Configuration

Centralized config for all models. Default = self-hosted, low-latency.
Cloud providers require explicit opt-in.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


def model_dir() -> Path:
    """Directory that model files live in and get downloaded to.

    FUSION_MODEL_DIR wins. Otherwise a `models/` folder beside the source
    tree is used if one exists (editable installs, existing dev setups),
    else the per-user cache. Never the current working directory, so the
    server finds its models no matter where it's started from.
    """
    env = os.getenv("FUSION_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    source_models = Path(__file__).resolve().parent.parent / "models"
    if source_models.is_dir():
        return source_models
    cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache / "fusion-runtime" / "models"


def model_dir_source() -> str:
    """Why model_dir() picked its directory, for showing to users."""
    if os.getenv("FUSION_MODEL_DIR"):
        return "FUSION_MODEL_DIR"
    if (Path(__file__).resolve().parent.parent / "models").is_dir():
        return "models/ folder next to the source code"
    return "user cache"


def resolve_model_path(path: str) -> Path:
    """Absolute paths are used as-is; anything else is relative to model_dir()."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else model_dir() / p


# Model paths are relative to model_dir() unless absolute (see resolve_model_path)
_LLM_MODEL_7B = "llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"  # split in 2; llama.cpp loads part 2 itself
_LLM_MODEL_SMALL = "llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"  # ~470MB, for 8GB machines
_TTS_MODEL = "tts/onnx/model.onnx"


class Provider(str, Enum):
    # STT
    FASTER_WHISPER = "faster_whisper"

    # LLM
    LLAMA_CPP = "llama_cpp"
    OPENAI = "openai"  # any OpenAI-compatible endpoint (api_base)

    # TTS
    KOKORO = "kokoro"

    # VAD
    SILERO = "silero"

    # Turn Detection
    PUNCTUATION = "punctuation"


class STTConfig(BaseModel):
    provider: Provider = Provider.FASTER_WHISPER
    model: str = "tiny.en"
    device: str = "auto"  # auto | cpu | cuda — auto falls back to cpu on Mac
    compute_type: str = "auto"  # auto | int8 | float16 — auto picks int8 on CPU
    language: Optional[str] = "en"
    beam_size: int = 1
    vad_filter: bool = True


class LLMConfig(BaseModel):
    provider: Provider = Provider.LLAMA_CPP
    model: str = _LLM_MODEL_7B
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = all
    n_batch: int = 512
    n_threads: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    # Streaming
    stream: bool = True
    # API providers
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class TTSConfig(BaseModel):
    provider: Provider = Provider.KOKORO
    model: str = _TTS_MODEL
    voice: str = "af_heart"
    sample_rate: int = 24000
    speed: float = 1.0
    # Streaming
    stream: bool = True
    chunk_size: int = 256


class VADConfig(BaseModel):
    provider: Provider = Provider.SILERO
    threshold: float = 0.5
    min_silence_ms: int = 100
    min_speech_ms: int = 250


class TurnDetectionConfig(BaseModel):
    provider: Provider = Provider.PUNCTUATION
    unlikely_threshold: float = 0.08
    # Two-tier trailing-silence requirement before a turn is considered
    # over, measured going *forward* from when the user actually stopped
    # talking (not from whenever the next STT result happens to arrive —
    # see PipelineOrchestrator._llm_stage's silence watcher for why that
    # distinction matters). The shorter threshold applies when the
    # accumulated transcript already looks sentence-complete
    # (TurnDetectorBase.looks_complete); the longer one is the safety net
    # for anything ambiguous — a trailed-off or mid-thought pause — so an
    # ordinary breath doesn't get mistaken for the end of the turn.
    min_confident_silence_ms: int = 200
    min_silence_ms: int = 700
    # Required *sustained* speech (per Silero VAD, while the bot is
    # generating/speaking) before treating it as a genuine interruption
    # rather than a noise blip or residual echo of the bot's own voice.
    barge_in_min_speech_ms: int = 300
    # Text-domain self-echo rejection: a candidate "user" turn is discarded
    # (never sent to the LLM) if a long enough run of its words appears
    # verbatim, in order, inside the text the bot itself most recently
    # spoke. This exists because the bot's own speaker bleed sometimes
    # slips past the client's mute/barge-in gating (a genuine barge-in
    # unmutes the mic before room reverb of the bot's *own* last sentence
    # has fully decayed) and gets transcribed as if it were new user
    # speech — see [[client-side-barge-in]]. Unlike acoustic echo
    # cancellation, this needs no audio reference signal at all: we
    # already know the bot's own words exactly, with zero noise, because
    # we generated them.
    #
    # `echo_min_match_words` is a floor so short, ordinary phrases (e.g.
    # a real "yes" or "computer" that happens to also appear in the bot's
    # last reply) can never match on their own — only a genuinely long
    # shared run trips this. `echo_containment_ratio` additionally
    # requires that run to cover most of what was said, not just a
    # fragment of a much longer, otherwise-unrelated utterance — so a
    # real follow-up question that merely reuses a couple of the bot's
    # words (ordinary topic continuity) is not discarded.
    echo_min_match_words: int = 4
    echo_containment_ratio: float = 0.5


class PipelineConfig(BaseModel):
    """Main pipeline configuration. Defaults to all self-hosted."""
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    turn_detection: TurnDetectionConfig = Field(default_factory=TurnDetectionConfig)
    
    # Global settings
    target_latency_ms: int = 500
    allow_cloud_fallback: bool = False  # Must explicitly enable
    enable_batching: bool = True
    batch_timeout_ms: int = 50
    max_batch_size: int = 8
    
    # Audio
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 20


# Default configs for different deployment profiles

DEVELOPMENT_CONFIG = PipelineConfig(
    stt=STTConfig(provider=Provider.FASTER_WHISPER, model="tiny.en", device="cpu", compute_type="int8"),
    llm=LLMConfig(
        provider=Provider.LLAMA_CPP,
        model=_LLM_MODEL_SMALL,
        n_gpu_layers=0,
        n_ctx=2048,
        max_tokens=256,
    ),
    tts=TTSConfig(provider=Provider.KOKORO, model=_TTS_MODEL, voice="af_heart"),
)

PRODUCTION_CONFIG = PipelineConfig(
    stt=STTConfig(provider=Provider.FASTER_WHISPER, model="tiny.en", device="cuda"),
    llm=LLMConfig(provider=Provider.LLAMA_CPP, model=_LLM_MODEL_7B, n_gpu_layers=-1),
    tts=TTSConfig(provider=Provider.KOKORO, model=_TTS_MODEL),
)

HYBRID_CONFIG = PipelineConfig(
    stt=STTConfig(provider=Provider.FASTER_WHISPER, model="tiny.en", device="cuda"),
    llm=LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", api_key="${OPENAI_API_KEY}"),
    tts=TTSConfig(provider=Provider.KOKORO, model=_TTS_MODEL),
    allow_cloud_fallback=True,
)