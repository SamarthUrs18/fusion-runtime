<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SamarthUrs18/fusion-runtime/main/docs/brand/logo-dark.png">
    <img alt="fusion-runtime" src="https://raw.githubusercontent.com/SamarthUrs18/fusion-runtime/main/docs/brand/logo-light.png" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/SamarthUrs18/fusion-runtime"><img alt="GitHub stars" src="https://img.shields.io/github/stars/SamarthUrs18/fusion-runtime?style=social&cacheSeconds=3600"></a>
  <a href="https://pypi.org/project/fusion-runtime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fusion-runtime?cacheSeconds=3600"></a>
  <a href="https://fusion-runtime.dev/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-fusion--runtime.dev-d9612f"></a>
  <img alt="Python 3.11 to 3.13" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
</p>

**A self-hosted voice agent runtime.** Speech-to-text, the LLM and text-to-speech run together on
one machine and stream into each other, so a reply starts playing while it's still being generated.

**On an RTX 3090 with a 7B model: about 490 ms of processing once a turn ends**, or 991 ms
stopwatched from your last syllable — the difference is a silence wait you can configure. 127
tokens/sec, interruptions honoured mid-sentence.

## Quickstart

Requires Python 3.11–3.13.

```bash
pip install fusion-runtime
```

An agent is one file. This is the whole thing:

```python
# agent.py
from fusion_runtime import Agent, LLM, STT, TTS, Turns

agent = Agent(
    name="shopkart-orders",
    prompt="You are the order line for ShopKart. Keep answers to one short sentence.",
    stt=STT("whisper-tiny.en"),          # or "whisper-small" for better accuracy
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=256),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=500, interrupt_after_ms=300),
)
```

```bash
frun models pull agent.py     # exactly the models it names, nothing else
frun up agent.py              # add --reload to restart on every edit
```

Then talk to it from a second terminal:

```bash
pip install "fusion-runtime[talk]"
frun talk
```

That is the whole loop — one file, two commands, a conversation. Talk over the agent to
interrupt it.

`frun up` with no file runs a default agent if you just want to hear it work, and `frun doctor`
checks libraries, GPU, models and audio and says how to fix what it finds.

### In a browser instead

The runtime serves a browser client at **http://localhost:8000** — the same one you would embed
in your own page.

With no keys configured, open it and click Talk. With keys configured (`FUSION_ACCEPTED_KEYS`),
a page can't hold a secret, so it needs a short-lived session token:

```bash
frun token        # prints a URL with a token in it — open that
```

Tokens are single-use and expire in about a minute. The page is handed its next one over the
socket it already has, so a conversation keeps going without asking again. If you open the bare
URL on a server with keys, the connection closes and the page says the token wasn't accepted.

### Naming models

A model is a catalog id (`frun models list`), a file path, `hf:owner/repo` for anything on
Hugging Face, or a URL for an OpenAI-compatible endpoint. A model on a vLLM, SGLang or
llama-server you started is named with that server in front — `vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ`
— and found at the server's usual address (`url=` for another).

Settings are checked against the runtime that runs the model, so a misspelt one fails at startup
with the name it probably meant instead of being ignored. For llama.cpp that includes
`flash_attn`, `kv_cache_type="q8_0"` (half the context memory), `use_mlock`, `main_gpu` and
`llama_kwargs` for anything else; for an endpoint, `extra_body` for server-specific sampling.

Secrets never go in the agent file — it names the *variable* holding a key
(`api_key_env="GROQ_API_KEY"`), so `agent.py` is safe to commit.

### Tools

A tool is a function. The model reads its name, docstring and type hints, calls it when it needs
to, and answers with what it returned:

```python
from fusion_runtime import Agent, LLM, tool

@tool
async def order_status(order_id: str) -> dict:
    """Look up where an order is and when it will arrive.

    Args:
        order_id: The order number, as the caller reads it out.
    """
    return await orders.lookup(order_id)

agent = Agent(prompt="...", llm=LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ"), tools=[order_status])
```

What the model says before calling ("Let me check.") is spoken while the tool runs. Each call has
a timeout (`@tool(timeout_s=...)`, 10 s by default); ordinary functions run on a worker thread;
a tool that fails tells the model what went wrong rather than ending the call; and talking over
the wait cancels it. After `max_tool_rounds` calls in one turn (4) the model has to answer.

Tools need an LLM server that can call them — vLLM (`--enable-auto-tool-choice
--tool-call-parser ...`), SGLang, llama-server (`--jinja`) or a hosted API. The in-process
llama.cpp runtime can't, and `frun up` says so at startup. A runnable version is
[`examples/tools_agent.py`](examples/tools_agent.py).

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
| **Processing** — turn ends, audio comes back | **~490 ms** | 288–657 |
| **Stopwatch from your last syllable** | **991 ms** | 858–1061 |
| ↳ of which: silence wait before the turn is judged over | ~500 ms | `turns.wait_ms` |
| Speech-to-text | 119 ms | 58–329 |
| LLM first token | 27 ms | 20–70 |
| First token → first audio (a sentence gets written, then spoken) | 430 ms | 320–509 |
| Text-to-speech real-time factor | 0.09 | speech is synthesized ~11× faster than real time |
| LLM tokens/sec | 127 | 106–130 |

**Two numbers, because there are two honest answers.** A stopwatch started at your last syllable
reads 991 ms. About 500 ms of that is the runtime waiting through silence to decide you've
finished — which elapses while you're still finishing, so people don't experience it as waiting.
What a caller feels is closer to the 490 ms of processing. Quote whichever you like, but say
which one: a voice stack claiming a number under 500 ms is almost always measuring from "we
decided the caller stopped", not "the caller stopped".

**The stages don't sum, and that's not sleight of hand.** Transcription of what you already said
runs during the silence wait. And "first token → first audio" is mostly the language model
writing a sentence — text-to-speech can't start on half a clause — so it is not a measure of how
fast Kokoro is. Kokoro's own speed is the real-time factor: 0.09, or about 126 ms of compute for
1.4 seconds of speech.

Every figure is the runtime's own per-turn telemetry (`frun talk --verbose`, or the browser
console), so you can reproduce them rather than trusting ours. Barge-in fired on every attempt.

## Several callers at once

Measured on the same 3090, real WebSocket sessions, three turns each
([`scripts/concurrency_check.py`](scripts/concurrency_check.py)). Response is the server's own
figure: the turn ends, audio comes back.

| Callers | In-process llama.cpp | vLLM | SGLang |
|---|---|---|---|
| 1 | ~460 ms | 398 ms | 410 ms |
| 4 | ~740 ms | 698 ms | 848 ms |
| 8 | ~4600 ms | 1086 ms | 1054 ms |
| 12 | ~7500 ms | 1083 ms | 1264 ms |
| 16 | — | 2064 ms | 1598 ms |

**With the model in-process, four callers is the ceiling.** At twelve, the language model's first
token takes 4790 ms of a 5312 ms response, while speech-to-text stays at 76 ms and text-to-speech
at 469 ms: one llama.cpp context decodes one reply at a time.

**With vLLM or SGLang, twelve callers answer in about a second.** Both batch every caller's reply
into each step on the GPU, and the language model's first token stayed between 36 and 72 ms from
one caller to sixteen. The limit moves to speech, which still runs one caller at a time: Kokoro's
first audio grows from ~350 ms alone to 1.5–2 s at sixteen callers, and Whisper grows with how
long people talk (a 3-second question: ~200 ms alone, 1.6–2.2 s at sixteen). Batching speech is
the next step, not the language model.

**Tools hold up under load.** Every caller asked about the same order on every turn: vLLM looked
it up in 122 of 123 turns, SGLang in 109. On SGLang every session looked it up the first time;
the turns without a lookup were the question asked a second, third or fourth time, answered from
the lookup already in the conversation. That is correct, but if your data can change during a
call, tell the agent to look things up again.

Measured 26 September 2026: Qwen2.5-7B-Instruct-AWQ, `--gpu-memory-utilization 0.6`,
`--max-model-len 4096`, Whisper small and Kokoro on the same card, both servers with default
settings otherwise. The in-process column is Qwen 7B q4 GGUF on llama.cpp, 21 September.

To run the model on a server and leave speech where it is:

```python
llm = LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ", url="http://localhost:8002/v1")
llm = LLM("llama_server:qwen2.5-7b-instruct")              # llama-server -np N, on :8080
llm = LLM("http://gpu-box:8000/v1", model_name="...")        # any OpenAI-compatible server
```

**On one 24 GB card** (RTX 3090, L4) the server shares the GPU with Whisper and Kokoro, and vLLM
reserves 90% of the card by default. Cap it, and start it first:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8002 \
  --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 16 \
  --enable-auto-tool-choice --tool-call-parser hermes
frun up agent.py        # an Agent(profile="production", ...), so Whisper runs on the GPU too
```

SGLang the same way (its usual port is 30000, and `sglang:` finds it there):

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct-AWQ --port 30000 \
  --mem-fraction-static 0.6 --context-length 4096 --tool-call-parser qwen25
```

`0.6` is about 14 GB: the weights plus every caller's context. Voice turns are short, so a 4096
context fits more callers than the model's maximum would. The tool flags are only needed for
tools, and the parser depends on the model family. Port 8002, because `frun up` is on 8000 and some hosts (Runpod's pod images) already use 8001.

A 4-bit model (AWQ or GPTQ) leaves room for speech; a 16-bit 7B model needs ~15 GB for its
weights alone and doesn't. The L4 has about a third of the 3090's memory bandwidth, so expect
slower tokens there. `nvidia-smi` shows what is actually used.

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

## Using it as a library

`frun up agent.py` covers running an agent. The pipeline can also run inside your own process —
for a queue worker, a test, or a batch job over recorded calls — with no server involved. See
[`examples/sdk_example.py`](examples/sdk_example.py), which is runnable, and
[the docs](https://fusion-runtime.dev/docs#python).

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
