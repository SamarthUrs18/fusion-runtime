# fusion-runtime

**A self-hosted voice agent runtime.** Speech-to-text, the LLM and text-to-speech run together on one machine and stream into each other, so a reply starts playing while it's still being generated.

> **Status: early development.** The full voice pipeline works on a laptop CPU, including interruptions and echo cancellation. It hasn't been measured on a GPU yet, and it serves one conversation at a time. See [Roadmap](#roadmap).

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/fusion-runtime/fusion-runtime.git
cd fusion-runtime
pip install -e ".[talk]"

frun models pull          # ~0.9 GB: Whisper tiny, Qwen2.5 0.5B, Kokoro, Silero VAD
frun doctor               # checks libraries, GPU, models and audio, and says how to fix problems
```

Start the server, then talk to it from a second terminal:

```bash
frun up
```

```bash
frun talk
```

On macOS, allow microphone access for your terminal app (System Settings → Privacy & Security → Microphone). You can talk over the bot to interrupt it. With headphones, `frun talk --no-aec` turns echo cancellation off.

## The `frun` CLI

| Command | What it does |
|---------|--------------|
| `frun up` | Starts the server on `127.0.0.1:8000`. Checks models are installed and the port is free first |
| `frun up --config production --host 0.0.0.0 --port 8080` | Production models, reachable from other machines |
| `frun talk` | Talks to the server with your mic and speakers |
| `frun talk --url ws://host:8080/v1/voice/ws` | Talks to a server elsewhere |
| `frun models list` | Shows every model, its size, whether it's installed, and which profile uses it |
| `frun models pull` | Downloads what the development profile needs |
| `frun models pull --config production` | Downloads what the production profile needs |
| `frun models pull --llm` | Only one stage; also `--whisper`, `--kokoro`, `--vad` |
| `frun models pull qwen2.5-7b-q4` | A specific model by ID |
| `frun doctor` | Checks Python, libraries (incl. that torch and torchaudio match and Silero VAD really loads), GPU support, models, port and audio. Prints a fix for each problem; exits 1 if something is broken |
| `frun version` | Installed version |

`fusion-runtime` works as an alias for `frun`. If the command isn't on your PATH, use `python3 -m fusion_runtime.cli`.

## Models

Every download is pinned to an exact Hugging Face commit and checked by file size ([catalog](fusion_runtime/catalog/models.toml)).

| ID | Stage | Size | License | Used by |
|----|-------|------|---------|---------|
| `whisper-tiny.en` | Speech-to-text | 78 MB | MIT | development, production |
| `qwen2.5-0.5b-q4` | LLM (GGUF) | 491 MB | Apache-2.0 | development |
| `qwen2.5-7b-q4` | LLM (GGUF) | 4.7 GB | Apache-2.0 | production |
| `kokoro-v1.0` | Text-to-speech (ONNX) | 328 MB | Apache-2.0 | development, production |
| `silero-vad` | Voice activity detection | 2 MB | MIT | development, production |

**Where models are stored**, first match wins:

1. `FUSION_MODEL_DIR`, if set
2. a `models/` folder next to the source code (source checkouts)
3. `~/.cache/fusion-runtime/models`

Silero VAD is the exception: it's cached by PyTorch in `~/.cache/torch/hub`.

## How it works

```
 microphone audio
       │
       ▼
 Silero VAD ──► faster-whisper ──► turn detection ──► llama.cpp ──► Kokoro ──► audio out
 (speech only)   (rolling window)   (punctuation +     (streams      (speaks each
                                     silence)           tokens)       sentence, or
                                                                      a long phrase)
       │
       └──► barge-in watcher: if you talk over the reply, generation and playback stop
```

Everything runs in one Python process today. The local client removes the bot's own voice from the microphone ([echo cancellation](fusion_runtime/audio/echo_canceller.py)), so the server hears clean audio and can decide when you're interrupting.

## Configuration

Choose a profile with `frun up --config` (or `FUSION_CONFIG` if you start the server another way):

| Profile | STT | LLM | TTS | For |
|---------|-----|-----|-----|-----|
| `development` (default) | Whisper tiny, CPU int8 | Qwen2.5 0.5B, CPU | Kokoro | Laptops, 8 GB RAM |
| `production` | Whisper tiny, CUDA | Qwen2.5 7B, all GPU layers | Kokoro | NVIDIA GPU |

```bash
frun up --config production
```

In Python, model paths are relative to the model directory:

```python
from fusion_runtime import PipelineConfig, LLMConfig

PipelineConfig(
    llm=LLMConfig(provider="llama_cpp", model="llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
)
LLMConfig(provider="openai", model="...", api_base="http://localhost:8080/v1")  # any OpenAI-compatible server
```

Only `FUSION_CONFIG`, `FUSION_MODEL_DIR` and `FUSION_AEC` (`FUSION_AEC=0` is the same as `frun talk --no-aec`) are read today. The other variables in `.env.example` aren't wired up yet.

## Server API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Status and loaded models |
| `POST /v1/voice/chat` | One turn: base64 audio in, base64 audio out |
| `POST /v1/voice/stream` | One turn, streamed PCM response |
| `WS /v1/voice/ws` | Real-time conversation: raw 16 kHz PCM in, 24 kHz PCM and JSON events out |
| `GET /metrics` | P50/P99 latency over recent turns |

There's no authentication yet, which is why `frun up` only listens on `127.0.0.1` unless you pass `--host`.

## Performance

Measured numbers only.

| Machine | Profile | Time to first audio | Speech-to-text |
|---------|---------|---------------------|----------------|
| 8 GB MacBook Air, CPU only | development | ~1.6 s | ~0.5–0.7 s |
| NVIDIA GPU | production | not measured yet | not measured yet |

Measured on the `tests/fixtures/hello.wav` clip.

## Development

```bash
pip install -e ".[dev,talk]"
pytest tests/
```

```
fusion_runtime/
├── cli/          frun commands, one file per command (_talk_client.py is the mic client)
├── catalog/      model catalog (models.toml), install checks, downloads
├── config.py     settings, profiles, model directory
├── server.py     FastAPI + WebSocket server
├── stt/          base.py, whisper.py
├── llm/          base.py, llama_cpp.py, openai_compat.py
├── tts/          base.py, kokoro.py
├── vad/          base.py, silero.py, turn.py
├── engine/       orchestrator.py (conversation loop), barge_in.py, metrics.py
└── audio/        echo_canceller.py, duplex_audio.py
```

`docker/` and `modal_deploy.py` exist but haven't been verified yet.

## Roadmap

- **Phase 1, in progress:** `frun` CLI (`models`, `up`, `talk`, `doctor` done; `deploy --target modal` next) and the first GPU measurements
- **Phase 2:** many conversations per GPU: LLM server sidecar, per-call sessions, admission control
- **Phase 3:** agents as Python files (`frun up agent.py`): prompts, variables, hooks and tool calling
- **Phase 4:** deploy environments (`fusion.toml`) and a model picker
- **Then:** browser/app client, phone calls, managed cloud

## License

AGPL-3.0-or-later, as declared in `pyproject.toml`. Licensing is not final yet.
