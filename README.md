<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.png">
    <img alt="fusion-runtime" src="docs/brand/logo-light.png" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/SamarthUrs18/fusion-runtime"><img alt="GitHub stars" src="https://img.shields.io/github/stars/SamarthUrs18/fusion-runtime?style=social"></a>
  <a href="https://fusion-runtime.dev/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-fusion--runtime.dev-d9612f"></a>
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
</p>

**A self-hosted voice agent runtime.** Speech-to-text, the LLM and text-to-speech run together on
one machine and stream into each other, so a reply starts playing while it's still being generated.

**On an RTX 3090 with a 7B model: 472 ms from the caller finishing to audio coming back**, 105
tokens/sec, interruptions honoured mid-sentence.

## Quickstart

Requires Python 3.11–3.13.

```bash
pip install "fusion-runtime[talk]"
frun models pull          # ~0.9 GB: Whisper tiny, Qwen2.5 0.5B, Kokoro, Silero VAD
frun up
```

Not on PyPI until the first release. Until then, from a clone:
`pip install -e ".[talk]"`.

Then open **http://localhost:8000** and click Talk. That page is served by the runtime itself —
no build step, nothing to install. You can talk over the agent to interrupt it.

`frun talk` does the same from a terminal. `frun doctor` checks libraries, GPU, models and audio,
and says how to fix what it finds.

## An agent is one file

```python
# agent.py
from fusion_runtime import Agent, LLM, STT, TTS, Turns

agent = Agent(
    name="shopkart-orders",
    prompt="You are the order line for ShopKart. Keep answers to one short sentence.",
    stt=STT("whisper-tiny.en"),
    llm=LLM("qwen2.5-7b-q4", max_tokens=200),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
)
```

```bash
frun models pull agent.py     # exactly the models it names
frun up agent.py --reload
```

A model is named as a catalog id (`frun models list`), a file path, `hf:owner/repo` for anything
on Hugging Face, or a URL for an OpenAI-compatible endpoint. Settings the config knows are
applied; anything else is passed through to that runtime.

Secrets never go in the agent file — it names the *variable* holding a key
(`api_key_env="GROQ_API_KEY"`), so `agent.py` is safe to commit.

## From your own Python

`frun up` is a thin wrapper. The same pipeline runs inside your process, so you can put a voice
turn behind a queue worker, a test, or a batch job with no server involved:

```python
import asyncio, wave
from fusion_runtime import Agent, LLM, STT, TTS, PipelineOrchestrator, run_single_turn

agent = Agent(
    prompt="You are the order line for ShopKart. Answer in one short sentence.",
    stt=STT("whisper-tiny.en"),
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=60),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
)

async def main():
    orchestrator = PipelineOrchestrator(agent.config())
    await orchestrator.initialize()          # loads the models once; reuse it across turns
    with wave.open("caller.wav", "rb") as w:
        audio = w.readframes(w.getnframes())
    reply = await run_single_turn(orchestrator, audio, system_prompt=agent.prompt)
    print(f"{len(reply) / 2 / 24000:.2f}s of speech")   # 24 kHz mono 16-bit PCM
    await orchestrator.shutdown()

asyncio.run(main())
```

`agent.config()` is the agent resolved against its profile and the environment — the same
`PipelineConfig` the server builds. `load_agent("agent.py")` returns the `Agent` from a file, so a
script and `frun up` can share one definition.

`run_single_turn` waits for the whole reply. For audio as it is produced — which is what makes
barge-in possible — use `orchestrator.run_pipeline(audio_chunks, prompt)`, an async iterator of
PCM chunks. `initialize()` is the expensive call; hold the orchestrator and reuse it.

## On your own site

```html
<script src="https://your-server/fusion-runtime.js"></script>
<button id="talk"></button>
<script>FusionRuntime.attach({ button: "#talk" });</script>
```

The runtime serves the browser client it uses itself, so the page you demo with is the one your
site embeds. With no `url` it connects back to wherever the script came from.

Browsers only allow a microphone on `https://`, so a deployment needs TLS and `wss://`. A page
never holds an API key: your backend mints it a short-lived token.

## Authentication

```bash
frun key new
FUSION_ACCEPTED_KEYS=web:frun_kR7m...
```

Without keys the server answers on `localhost` only, and `frun up --host 0.0.0.0` refuses to
start. `frun talk`, a backend or curl send the key in an `Authorization` header; a browser page
gets a short-lived, single-use token from `POST /v1/sessions` instead, because a page can hold
neither a secret nor a header.

Concurrency caps, message and audio limits, idle timeouts, origin allowlists and proxy trust all
have working defaults — see the docs.

## Performance

Measured, not estimated. RTX 3090, Qwen 7B q4 + Whisper tiny.en + Kokoro, all on the one card:

| | Median | Range |
|---|---|---|
| **Response** — caller stops, audio comes back | **472 ms** | 169–706 |
| LLM tokens/sec | ~105 | 92–122 |
| Text-to-speech real-time factor | 0.14 | 0.04–0.25 |

Barge-in fired on six of six attempts. Every figure comes from the runtime's own per-turn
telemetry (`frun talk --verbose`), so you can reproduce them rather than trusting ours.

A 7B answers about as fast as a 0.5B did on an 8 GB MacBook Air, because the end-of-turn wait and
the first sentence of speech dominate — not the model. Not yet measured: several callers at once,
and long calls.

## Concurrency

Conversations are fully isolated. For several at once, point the LLM at vLLM or `llama-server` —
the runtime already speaks to both — since the in-process model decodes one reply at a time:

```python
llm = LLM("http://localhost:8080/v1", model_name="qwen2.5-7b-instruct")
```

Shared-model scaling for speech-to-text is the next piece of work.

## The `frun` CLI

| | |
|---|---|
| `frun up [agent.py]` | Starts the server. `--host`, `--port`, `--reload`, `--config` |
| `frun talk` | Talks to it from a terminal, with a latency summary per turn |
| `frun models list` / `pull` | What's available, and downloading it |
| `frun key new` / `keys list` / `token` | Keys and browser tokens |
| `frun doctor` | Checks the machine and says how to fix what's wrong |

`fusion-runtime` works as an alias for `frun`.

## Documentation

Everything else — configuration, turn detection, languages, the server API, telemetry, limits,
GPU setup and deployment — is at **[fusion-runtime.dev/docs](https://fusion-runtime.dev/docs)**.

## Development

```bash
uv sync --extra dev --extra talk     # or: pip install -e ".[dev,talk]"
pytest
```

CI runs the suite on Python 3.11, 3.12 and 3.13. [CONTRIBUTING.md](CONTRIBUTING.md) has the
layout, the design rules a review will hold you to, and how to add a runtime.

## License

Not decided yet. `pyproject.toml` currently declares AGPL-3.0-or-later; that is under review and
will be settled before the first public release.
