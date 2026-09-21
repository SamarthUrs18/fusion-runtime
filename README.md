<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/logo-dark.png">
    <img alt="fusion-runtime" src="docs/brand/logo-light.png" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/SamarthUrs18/fusion-runtime"><img alt="GitHub stars" src="https://img.shields.io/github/stars/SamarthUrs18/fusion-runtime?style=social"></a>
  <a href="https://pypi.org/project/fusion-runtime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fusion-runtime"></a>
  <a href="https://fusion-runtime.dev/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-fusion--runtime.dev-d9612f"></a>
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
</p>

**A self-hosted voice agent runtime.** Speech-to-text, the LLM and text-to-speech run together on
one machine and stream into each other, so a reply starts playing while it's still being generated.

**On an RTX 3090 with a 7B model: about 1 second from the caller finishing speaking to audio
coming back** — roughly half of it a silence wait you can configure — at 127 tokens/sec, with
interruptions honoured mid-sentence.

## Quickstart

Requires Python 3.11–3.13.

```bash
pip install fusion-runtime
frun models pull          # ~0.9 GB: Whisper tiny, Qwen2.5 0.5B, Kokoro, Silero VAD
frun up
```

Not on PyPI until the first release. Until then, from a clone: `pip install -e .`.

Then open **http://localhost:8000** and click Talk. That page is served by the runtime itself —
no build step, nothing to install. You can talk over the agent to interrupt it.

`frun talk` does the same from a terminal — that one needs a microphone library, so
`pip install "fusion-runtime[talk]"`. `frun doctor` checks libraries, GPU, models and audio, and
says how to fix what it finds.

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
PCM chunks, roughly one per sentence. `initialize()` is the expensive call; hold the orchestrator
and reuse it.

[`examples/sdk_example.py`](examples/sdk_example.py) runs both paths against a real recording and
writes the reply to a WAV file:

```bash
frun models pull
python3 examples/sdk_example.py            # or: python3 examples/sdk_example.py my-recording.wav
```

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

Measured, not estimated. The production profile as it ships — RTX 3090, Qwen 7B q4 + Whisper
small + Kokoro on the one card — through the browser client, 21 September 2026:

| | Median | Range |
|---|---|---|
| **Time to first audio** — you stop speaking, you hear a reply | **991 ms** | 858–1061 |
| Speech-to-text | 119 ms | 58–329 |
| LLM first token | 27 ms | 20–70 |
| Text-to-speech, first chunk | 430 ms | 320–509 |
| LLM tokens/sec | 127 | 106–130 |

**Read the first row carefully, because it is the one that gets misquoted.** 991 ms is what a
caller lives through. Most of the gap between it and the stages below is the runtime deliberately
waiting through silence to decide the caller has finished (`turns.wait_ms`, 500 ms by default) —
a setting, not a speed limit. The stages don't sum to the total because they overlap: transcription
of what you already said runs during that wait. And the language model is 27 ms of it, which is
usually the opposite of where people expect a voice agent's time to go.

If you see a smaller number quoted for a voice stack, check whether it starts at "the caller
stopped talking" or at "we decided the caller stopped talking". Those differ by about half a
second, and only the first one is a caller's experience.

Every figure is the runtime's own per-turn telemetry (`frun talk --verbose`, or the browser
console), so you can reproduce them rather than trusting ours. Barge-in fired on every attempt.

## Several callers at once

Measured on the same 3090, real WebSocket sessions, three turns each:

| Callers | Response, median | Turns/sec |
|---|---|---|
| 1 | ~460 ms | 0.21 |
| 4 | ~740 ms | 0.55 |
| 8 | ~4600 ms | 0.69 |
| 12 | ~7500 ms | 0.74 |

**Four simultaneous callers land in the same range as one**, within run-to-run variance. Past
that it saturates: throughput plateaus around 0.7 turns/sec, so an extra caller past the knee
buys queue time rather than capacity. Eight is not a conversation.

The bottleneck is one specific thing. At twelve callers the language model's first token takes
4790 ms of a 5312 ms response, while speech-to-text stays at 76 ms and text-to-speech at 469 ms.
A single in-process llama.cpp context decodes one reply at a time; the speech stages do not care
how many callers there are.

So to go past four, move the language model out and leave speech where it is:

```python
llm = LLM("http://localhost:8080/v1", model_name="qwen2.5-7b-instruct")
```

vLLM and `llama-server -np N` both speak the API the `openai_http` runtime uses. Whether that
moves the knee, and how far, is not yet measured.

## The `frun` CLI

| | |
|---|---|
| `frun up [agent.py]` | Starts the server. `--host`, `--port`, `--reload`, `--config` |
| `frun talk` | Talks to it from a terminal, with a latency summary per turn |
| `frun models list` / `pull` | What's available, and downloading it |
| `frun key new` / `keys list` / `token` | Keys and browser tokens |
| `frun doctor` | Checks the machine and says how to fix what's wrong |
| `frun version` | The installed version. `--version` and `-V` work too |

`fusion-runtime` works as an alias for `frun`.

## Documentation and contact

Everything else — configuration, turn detection, languages, the server API, telemetry, limits,
GPU setup and deployment — is at **[fusion-runtime.dev/docs](https://fusion-runtime.dev/docs)**.

| | |
|---|---|
| Site and docs | **[fusion-runtime.dev](https://fusion-runtime.dev)** |
| Questions, or anything else | **hello@fusion-runtime.dev** |
| Security problems | **security@fusion-runtime.dev** — not a public issue, please ([why](CONTRIBUTING.md#security)) |

## Development

```bash
uv sync --extra dev --extra talk     # or: pip install -e ".[dev,talk]"
pytest
```

CI runs the suite on Python 3.11, 3.12 and 3.13. [CONTRIBUTING.md](CONTRIBUTING.md) has the
layout, the design rules a review will hold you to, and how to add a runtime.

## License

[Apache-2.0](LICENSE). Embed it in a commercial product, rebrand it, ship it closed — keep the
copyright notice and the `NOTICE` file in what you distribute, and don't use the project's name
to imply it endorses you.

The models it downloads by default are permissive too (Whisper MIT, Silero VAD MIT, Qwen2.5
Apache-2.0, Kokoro Apache-2.0), so the whole default path is clear for commercial use. A model
you point it at yourself carries its own licence — check that one before you ship it.
