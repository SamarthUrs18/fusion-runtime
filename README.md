# fusion-runtime

**A self-hosted voice agent runtime.** Speech-to-text, the LLM and text-to-speech run together on one machine and stream into each other, so a reply starts playing while it's still being generated.

> **Status: early development.** The full voice pipeline works on a laptop CPU, including interruptions and echo cancellation. It hasn't been measured on a GPU yet. Conversations are fully isolated; for several at once, point the LLM at vLLM or `llama-server` — the runtime already speaks to both — since the in-process model decodes one reply at a time. Shared-model scaling for speech-to-text is the next piece of work. See [Roadmap](#roadmap).

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/fusion-runtime/fusion-runtime.git
cd fusion-runtime
pip install -e ".[talk]"

frun models pull          # ~0.9 GB: Whisper tiny, Qwen2.5 0.5B, Kokoro, Silero VAD
frun doctor               # checks libraries, GPU, models and audio, and says how to fix problems
```

Start the server:

```bash
frun up
```

Then open **http://localhost:8000** and click Talk. That page is served by the runtime itself — no build step, nothing to install. You can talk over the agent to interrupt it.

Or talk from a second terminal instead:

```bash
frun talk
```

On macOS, allow microphone access for your browser (or, for `frun talk`, your terminal app) under System Settings → Privacy & Security → Microphone. With headphones, `frun talk --no-aec` turns echo cancellation off.

## Build an agent

An agent is one Python file: what it says, which models it uses, how it takes turns.

```python
# agent.py
from fusion_runtime import Agent, LLM, STT, TTS, Turns, VAD

agent = Agent(
    name="shopkart-orders",
    prompt="You are the order line for ShopKart. Keep answers to one short sentence.",
    stt=STT("whisper-tiny.en"),
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=200),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=500, interrupt_after_ms=300),   # answer after 500 ms of silence
    vad=VAD(threshold=0.5),                             # what counts as speech
)
```

```bash
frun up agent.py --reload     # --reload restarts when you edit the file
frun talk                     # in another terminal
```

| Part | What it sets |
|---|---|
| `STT` / `LLM` / `TTS` | The model for that stage, plus its settings. A plain name works when there's nothing to configure: `llm="qwen2.5-0.5b-q4"` |
| `Turns` | When the agent answers (`wait_ms`), when it stops for the caller (`interrupt_after_ms`), and how soon speaking again continues the same turn (`resume_window_ms`). A turn detector model goes first: `Turns("my_pkg.turns:MyDetector", wait_ms=400)` |
| `VAD` | Which audio counts as speech (`threshold`), feeding both of the above |

A model is named as a catalog id (`frun models list`), a file path, `hf:owner/repo`, or a URL,
optionally with the runtime in front (`vllm:hf:org/model`). Settings the config knows
(`max_tokens`, `voice`, `n_ctx`, ...) are applied; anything else is passed to that runtime.

**Any Hugging Face model works**, not just the catalog:

```bash
frun models pull hf:Systran/faster-whisper-small     # into the model directory
export HF_TOKEN=hf_...                               # only for gated or private models
```

A server also downloads what its agent names, so a fresh machine needs no separate step
(`FUSION_AUTO_DOWNLOAD=0` turns that off). Downloads are checked against free disk space first,
and only the files needed to run the model are fetched.

**Settings can come from three places, and the later one wins:** the agent file, then
environment variables (`FUSION_TURN_WAIT_MS`, `FUSION_LLM_URL`, ...), then CLI flags. So a
deployment can change behaviour without editing the agent, and a flag is for a quick experiment.
Secrets are never in the agent file: those are environment variables (`api_key_env`, `HF_TOKEN`).

Working example: [examples/agent.py](examples/agent.py). Tool calling isn't built yet, so
`tools=` raises a clear error.

## Put it on a website

The runtime serves the browser client it uses itself, so a page needs two lines and no build step:

```html
<script src="https://your-server/fusion-runtime.js"></script>
<button id="talk"></button>
<script>FusionRuntime.attach({ button: "#talk" });</script>
```

With no `url`, it connects back to the server the script came from. Everything else is optional:

```js
const session = FusionRuntime.attach({ button: "#talk", token: sessionToken });

session.on("transcript", msg => { if (msg.is_final) show("You: " + msg.text); });
session.on("response",   msg => { if (msg.is_final) show(msg.text); });
session.on("trace",      msg => console.log(msg.summary));   // TTFA, tokens/sec, per-stage times
session.on("error",      e   => show(e.message));
```

`FusionRuntime.connect(options)` returns the same session without binding a button, for a page
that has its own controls (`session.start()`, `session.stop()`, `session.interrupt()`).

The client uses the browser's own echo canceller, resamples the microphone in an AudioWorklet so a
busy page can't stutter the audio, and schedules replies slightly ahead of real time so network
jitter doesn't leave gaps. Interruptions are decided on the server, which hears clean audio: when
it says the caller interrupted, queued audio is dropped immediately, including the tail of a
sentence already synthesized.

**Two things to know before deploying it.** Browsers only hand over a microphone on `https://`
(or `localhost`), so the page and the WebSocket both need TLS — `https://` and `wss://`. And a page
never holds an API key: your backend mints a short-lived `token` for it, which is what `token:`
above is for. See [Authentication](#authentication).

Working example: [examples/website.html](examples/website.html).

## Authentication

Generate a key, and put it where the server runs:

```bash
frun key new
```

```bash
FUSION_ACCEPTED_KEYS=web:frun_kR7m…      # .env, or the environment
```

**Without keys the server answers on localhost only**, and `frun up --host 0.0.0.0` refuses to
start. That isn't a warning you can dismiss: a runtime reachable from elsewhere with no
authentication is someone else's GPU, on your bill.

Two kinds of client, because only one of them can keep a secret:

| Client | Presents |
|---|---|
| `frun talk`, your backend, curl | the key: `Authorization: Bearer <key>` (or `frun talk --key`) |
| a browser page | a session token its backend minted — never a key |

On your own machine you only set `FUSION_ACCEPTED_KEYS`: talking to a server on `localhost`,
`frun talk` and `frun token` use a key it accepts. `FUSION_API_KEY` is for reaching a server
somewhere else, and is never used as a fallback for a remote address — that would send your
server's key to someone else's.

```bash
curl -X POST https://your-server/v1/sessions -H "Authorization: Bearer $FUSION_API_KEY"
# {"token":"…","expires_in":60,"ws_url":"wss://your-server/v1/voice/ws?token=…"}
```

The token works once and expires in about a minute, so a leaked URL, screenshot or log line is
worthless by the time anyone reads it. For your own machine, `frun token` prints a console URL
with one in it.

**Rotating and revoking.** `FUSION_ACCEPTED_KEYS` takes a list, so a rotation is: add the new key,
move clients across, drop the old one. Point `FUSION_ACCEPTED_KEYS_FILE` at a file instead and
`kill -HUP` re-reads it without restarting — which matters on a GPU, where a restart reloads the
models. Removing a key is complete: its tokens are dropped and its conversations are closed.

`frun keys list` shows names and fingerprints, never keys; the same fingerprint appears in the
logs, so you can tell which key is busy or failing.

### Limits

Authentication says who may use the server; these say how much of it one caller may take. All
optional, defaults shown.

| Variable | Default | Caps |
|---|---|---|
| `FUSION_MAX_SESSIONS` | 4 | conversations at once |
| `FUSION_MAX_SESSIONS_PER_KEY` | all of it for one key; all but one when keys are shared | per key |
| `FUSION_MAX_MESSAGE_BYTES` | 1 MB | one WebSocket message |
| `FUSION_MAX_TURN_AUDIO_S` | 60 | speech without a pause |
| `FUSION_MAX_SESSION_S` | 900 | one conversation |
| `FUSION_IDLE_TIMEOUT_S` | 60 | a socket that went quiet |
| `FUSION_CONNECTIONS_PER_MINUTE` | 30 | new sockets, and failed keys, per address |
| `FUSION_TOKENS_PER_MINUTE` | 600 | tokens one key may mint |

### Origins and proxies

`FUSION_ALLOWED_ORIGINS=https://shopkart.example` lists the websites whose pages may open a
socket. Unset means only pages this server itself serves — browsers do not stop one site
connecting to another, so the server checks. Clients that aren't browsers send no `Origin` and are
unaffected.

Behind a proxy that terminates TLS (RunPod's, nginx, Cloudflare), name it:

```bash
FUSION_TRUSTED_PROXY=10.0.0.0/8      # or =1 when nothing else can reach the port
```

Until you do, its `X-Forwarded-*` headers are ignored — anyone who can reach the port can write
those headers — and `/v1/sessions` will refuse to mint a token over what looks like plain HTTP.
The forwarded address is used for counting only, never to decide who may in.

## The `frun` CLI

| Command | What it does |
|---------|--------------|
| `frun up [agent.py]` | Starts the server on `127.0.0.1:8000`. Checks models are installed and the port is free first |
| `frun up --config production --host 0.0.0.0 --port 8080` | Production models, reachable from other machines |
| `frun up --log-format json` | One JSON log line per event, for deployments and log collectors (see [Observability](#observability)) |
| `frun talk` | Talks to the server with your mic and speakers, with a one-line latency summary per turn |
| `frun talk --key <key>` | ...to a server with authentication on. Also: `FUSION_API_KEY` |
| `frun key new [name]` | Generates a key. Shown once — nothing stores it |
| `frun keys list` | The configured keys: names and fingerprints, never the keys |
| `frun token` | Mints a session token and prints a console URL to open |
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

The voice detector goes in `torch-hub/` inside the same directory, so one directory — and on a
deployment one volume — holds everything a server needs. A copy already in PyTorch's own cache
is adopted rather than downloaded again; `TORCH_HOME` overrides the location.

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

### Settings in a `.env` file

Rather than exporting variables by hand, put them in `.env` next to your project (git ignores it,
and `.env.example` lists what it can hold):

```
HF_TOKEN=hf_...
FUSION_TURN_WAIT_MS=800
FUSION_LLM_URL=http://localhost:8080/v1
FUSION_LLM_MODEL=my-model
```

`frun` reads it from the directory you run in, and from the directory your agent file lives in.
Anything already exported in your shell wins, so a deployment's real environment is never
overwritten. Values are never logged — only the names of what was loaded.

Also read: `FUSION_CONFIG`, `FUSION_MODEL_DIR`, `FUSION_LOG_FORMAT`, `FUSION_LOG_LEVEL`, `FUSION_LOG_CONTENT` (set by `frun up`'s log flags) and `FUSION_AEC` (`FUSION_AEC=0` is the same as `frun talk --no-aec`). The other variables in `.env.example` aren't wired up yet.

## Server API

| Endpoint | Purpose |
|----------|---------|
| `GET /` | The console: open it in a browser and talk to the agent |
| `GET /fusion-runtime.js` | The browser client, for your own pages (see [Put it on a website](#put-it-on-a-website)) |
| `POST /v1/sessions` | Mints a browser's session token. Needs the key |
| `GET /health` | Status and loaded models. Open: load balancers need it |
| `POST /v1/voice/chat` | base64 audio in; audio out plus every turn's text and metrics (`transcript`, `response_text`, `turns[].user/.agent/.outcome/.metrics`) |
| `POST /v1/voice/stream` | One turn, streamed PCM response |
| `WS /v1/voice/ws` | Real-time conversation: raw 16 kHz PCM in, 24 kHz PCM and JSON events out. The first message gives both rates |
| `GET /metrics` | Prometheus metrics (see [Observability](#observability)) |
| `GET /v1/metrics/summary` | P50/P99 of recent single-shot runs, as JSON |

Everything except `/health`, `/` and `/fusion-runtime.js` needs a key — or, on a WebSocket, a session token. See [Authentication](#authentication).

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

Either tool works; both read the same `pyproject.toml`.

```bash
uv sync --extra dev --extra talk     # creates .venv and installs the locked versions
```

```bash
pip install -e ".[dev,talk]"         # into whatever Python you are using
```

```bash
pytest tests/
```

`uv.lock` pins every version, which is why it is committed: the torch and
torchaudio releases have to match exactly, and drifting apart once broke voice
detection silently for days. Run `uv lock` after changing a dependency.
`.python-version` puts new checkouts on 3.11, the oldest version supported; CI runs the
suite on 3.11, 3.12 and 3.13. 3.14 is not supported: `kokoro-onnx` doesn't publish for it,
so `pyproject.toml` caps there rather than letting the install fail halfway.

First install takes a while whichever tool you use: `llama-cpp-python` is
published as source only, so it compiles (a few minutes, and it needs a C++
toolchain — on macOS, Xcode command line tools).

### On an NVIDIA GPU

torch on Linux already carries CUDA — its PyPI wheels bundle the NVIDIA
libraries — so a normal install is the GPU one. Two packages do need help:
`onnxruntime` has a separate GPU name, and `llama-cpp-python` ships source only
and builds without CUDA unless told otherwise.

```bash
pip install -e ".[cuda]"                                                        # onnxruntime-gpu
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# or, to compile it yourself (needs the CUDA toolkit):
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

The `cu124` index is an older CUDA line that stops at torch 2.6, not a GPU
variant of the current one — installing our pinned 2.9.1 from it fails.

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

### Deploying

Any machine with an NVIDIA GPU runs this — there's nothing provider-specific in the runtime.
**RunPod is what the docs are written against**, because it was the cheapest of those checked, a
pod has no request timeout, and its proxy gives you `https://` and `wss://` without you handling a
certificate. Browsers refuse a microphone without those, so for a voice agent on a website it
saves the most work.

The shape of it: build the image, push it to a registry, start a pod from it with a volume
mounted at `FUSION_MODEL_DIR`, and set three variables — `FUSION_ACCEPTED_KEYS`,
`FUSION_ALLOWED_ORIGINS` and `FUSION_TRUSTED_PROXY` (see [Authentication](#authentication)).

`docker/` holds a Dockerfile that predates most of this runtime and has never been built. It is
being rewritten, and until then there is no verified deployment path. No latency number is
published for a GPU either, for the same reason: nobody has run one.


## License

AGPL-3.0-or-later, as declared in `pyproject.toml`. Licensing is not final yet.
