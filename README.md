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
| `frun up --log-format json` | One JSON log line per event, for deployments and log collectors (see [Observability](#observability)) |
| `frun talk` | Talks to the server with your mic and speakers, with a one-line latency summary per turn |
| `frun talk --verbose` | Also prints each turn's full timeline |
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
| `hybrid` | Whisper tiny | `gpt-4o-mini` over HTTP (key from `$OPENAI_API_KEY`) | Kokoro | Speech local, LLM elsewhere |

```bash
frun up --config production
```

### LLM from an OpenAI-compatible endpoint

Speech stays local; the LLM comes from vLLM, llama-server, Ollama or a hosted API:

```bash
frun up --llm-url http://localhost:8080/v1 --llm-model my-model
export GROQ_API_KEY=...        # the key stays in the environment
frun up --llm-url https://api.groq.com/openai/v1 --llm-model <model> --llm-api-key-env GROQ_API_KEY
```

The same settings as environment variables: `FUSION_LLM_URL`, `FUSION_LLM_MODEL`, `FUSION_LLM_API_KEY_ENV`.
API keys are never accepted in config, only the *name* of the variable holding one. `frun doctor` checks
that the key is set, the endpoint answers and it serves the model.

### In Python

A stage names its runtime and any model reference (catalog id, path, `hf:owner/repo`, URL):

```python
from fusion_runtime import PipelineConfig, LLMConfig, STTConfig

PipelineConfig(
    stt=STTConfig(model="whisper-tiny.en"),
    llm=LLMConfig(runtime="llama_cpp", model="qwen2.5-0.5b-q4", n_ctx=2048),
)
LLMConfig(runtime="openai_http", model="http://localhost:8080/v1",
          api_key_env="MY_KEY", options={"model_name": "my-model"})
LLMConfig(runtime="my_package.llm:MyRuntime", model="anything")  # a plugin runtime
```

### Turn detection: when the agent answers

By default the agent answers after **500 ms of silence**, and stops speaking once the caller has
talked over it for **300 ms**. Both are settings:

```bash
frun up --turn-wait-ms 800          # more patient with callers who pause mid-sentence (FUSION_TURN_WAIT_MS)
frun up --interrupt-after-ms 500    # ignore coughs and "mm-hm" in noisy places (FUSION_INTERRUPT_AFTER_MS)
```

In Python: `TurnDetectionConfig(min_silence_ms=..., barge_in_min_speech_ms=..., resume_window_ms=...)`.

- **Speaking again right after a pause continues the same turn.** If the caller cuts off the reply
  within 1.5 s of their turn ending (`resume_window_ms`), both parts reach the LLM as one message:
  "hello ... my name is Priya" is answered once, not twice.
- **Turn detector models plug in.** A detector predicts whether the caller is done, from the words,
  the conversation or the audio. The wait then shortens to `min_confident_silence_ms` when they're
  likely done, and stretches to `max_silence_ms` when they're likely mid-thought. Silence always
  confirms the end of a turn, and a detector that's slow or fails just leaves the default wait.

```bash
frun up --turn-detector my_package.turns:MyDetector    # or a plugin name (FUSION_TURN_DETECTOR)
```

Write one by implementing `fusion_runtime.contract.TurnDetector.predict`, and check it with the
conformance kit: see [examples/turn_detector_plugin.py](examples/turn_detector_plugin.py). No
detector model is bundled yet; use any model whose license allows your use.

### Languages

Set what callers speak with `STTConfig(language="hi")` (`None` detects it per turn; English-only
Whisper models like `tiny.en` refuse other languages at startup). Kokoro speaks its voice's language,
or `TTSConfig(language=...)`. Reply text is split into sentences for speech in any script (`.` `।` `。`
`؟` …), and decimals like "3.5" aren't cut. The catalog currently has English models only.

Also read: `FUSION_CONFIG`, `FUSION_MODEL_DIR`, `FUSION_LOG_FORMAT`, `FUSION_LOG_LEVEL`, `FUSION_LOG_CONTENT` (set by `frun up`'s log flags) and `FUSION_AEC` (`FUSION_AEC=0` is the same as `frun talk --no-aec`). The other variables in `.env.example` aren't wired up yet.

## Server API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Status and loaded models |
| `POST /v1/voice/chat` | One turn: base64 audio in, base64 audio out |
| `POST /v1/voice/stream` | One turn, streamed PCM response |
| `WS /v1/voice/ws` | Real-time conversation: raw 16 kHz PCM in, 24 kHz PCM and JSON events out |
| `GET /metrics` | Prometheus metrics (see [Observability](#observability)) |
| `GET /v1/metrics/summary` | P50/P99 of recent single-shot runs, as JSON |

There's no authentication yet, which is why `frun up` only listens on `127.0.0.1` unless you pass `--host`.

WebSocket clients receive, besides audio: `transcript`, `response`, `interrupted`, `echo_discarded`, `turn.trace` (every turn's summary and timeline) and `error` (`code`, `message`, `stage`, `retryable`, `fix`; never a stack trace).

## Observability

Every stage emits structured events with a timestamp, session id, turn id, stage, model and duration, so a deployed agent never runs blind.

```
20:47:49.441  95b134d2 t1   vad      speech_start       audio_offset_ms=192 probability=0.88
20:47:52.162  95b134d2 t1   turn     end_detected       reason=silence detector=silence threshold_ms=500 wait_ms=569
20:47:52.334  95b134d2 t1   llm      first_token        171ms runtime=llama_cpp model=llm/qwen2.5-0.5b-instruct-q4_k_m.gguf
20:47:52.670  95b134d2 t1   tts      first_chunk        326ms audio_ms=2525
20:47:52.671  95b134d2 t1   audio    first_sent         response_ms=509 ttfa_ms=778
20:47:55.980  95b134d2 t1   turn t1 completed · TTFA 778ms · response 509ms · end-of-turn wait 269ms · llm first 171ms, 91 tok/s · tts first 337ms, rtf 0.24
```

- **Logs:** `frun up --log-format pretty|json`, `--log-level debug|info|warning|error`. Errors include a stable `code`, whether retrying can help, a suggested fix, and the stack trace (server logs only).
- **Private by default:** logs record text *lengths*, not what people said. `--log-content` includes transcripts and replies. API keys are always redacted.
- **Per turn:** time to first audio (TTFA, from the end of the user's speech), end-of-turn wait, transcription delay, speech-to-text time, LLM time to first token and tokens per second, TTS time to first audio and real-time factor, playback start, interruptions and how fast generation stopped. Speech timings use when audio *arrived*, so they stay correct when the server is busy.
- **`GET /metrics` (Prometheus):** latency histograms for the numbers above, turns by outcome, errors by stage and code, active and ended sessions, loaded models and load times, interruptions, echo rejections, event-loop lag and stalls, process memory, CPU and threads, and GPU memory when CUDA is in use.
- **Event-loop monitor:** warns whenever something blocks the server's event loop for over 100 ms, since that freezes audio input and interruptions for every conversation.

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
├── catalog/      model catalog (models.toml), install checks, downloads, GGUF metadata
├── config.py     settings, profiles, model directory
├── server.py     FastAPI + WebSocket server
├── contract/     the interface every model runtime implements (STT, LLM, TTS)
├── resolver.py   model reference (catalog id, path, hf:repo, URL) → runtime
├── registry.py   runtime names and plugins → classes
├── runtimes/     one adapter per engine, not per model:
│                 llama_cpp (any GGUF), ctranslate2 (Whisper), onnx (+ Kokoro spec), openai_http
├── engine/       orchestrator.py (conversation loop), scheduler.py, conversation.py,
│                 streaming.py (rolling STT), text.py (sentences), barge_in.py, metrics.py
├── vad/          base.py, silero.py, turn.py
├── telemetry/    events, logs, Prometheus metrics, per-turn traces
├── testing/      conformance kit and fake runtimes
└── audio/        echo_canceller.py, duplex_audio.py
```

`docker/` and `modal_deploy.py` exist but haven't been verified yet.


## License

AGPL-3.0-or-later, as declared in `pyproject.toml`. Licensing is not final yet.
