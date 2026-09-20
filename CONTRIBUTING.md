# Contributing

Thanks for looking. This file is for people changing the code. If you want to *use*
fusion-runtime, the [README](README.md) and [fusion-runtime.dev/docs](https://fusion-runtime.dev/docs)
are the right places.

## Before you start

**The licence isn't settled.** `pyproject.toml` currently declares AGPL-3.0-or-later, and that is
under review until the first public release. If the licence matters to how you can use or
contribute to this, wait for the release rather than assuming the current declaration is final.

**Don't paste code from other voice frameworks.** LiveKit, Pipecat and the rest are fine to read
and learn from; their code, their packages and their model weights carry licences this project
hasn't accepted. A contribution that copies from them can't be merged, and it's much cheaper to
catch that before you write it than in review.

## Setup

```bash
uv sync --extra dev --extra talk     # or: pip install -e ".[dev,talk]"
frun models pull                     # ~0.9 GB, needed for the integration tests
pytest
```

`uv.lock` is committed on purpose. torch and torchaudio have to match exactly; when they drifted
apart, voice activity detection broke *silently* and stayed broken for days before anyone noticed.
If you change a dependency, commit the regenerated lock file with it.

## Tests

```bash
pytest                       # everything
pytest -m "not integration"  # what CI runs — no model weights needed
pytest -m integration        # the real models, end to end
```

CI runs the non-integration suite on Python 3.11, 3.12 and 3.13. Integration tests skip
themselves when the models aren't downloaded, which is why CI stays fast and why a green CI run
doesn't prove a model change works — run them locally for anything touching a runtime.

`ruff` and `mypy` are configured (line length 100) but not enforced in CI. Run them before you
open a pull request.

## How the code is laid out

```
fusion_runtime/
├── cli/          frun commands, one file per command
├── catalog/      model catalog, install checks, downloads
├── contract/     the interface every model runtime implements
├── runtimes/     one adapter per engine, not per model
├── engine/       orchestrator, scheduler, streaming, barge-in
├── security/     keys, session tokens, limits, origins
├── web/          the console and the browser client
├── vad/  turns/  telemetry/  audio/  testing/
```

## Rules a review will hold you to

These aren't style preferences. Each one is here because breaking it cost real debugging time.

**Adapters go per runtime, not per model.** `llama_cpp` runs any GGUF because the architecture and
chat template are read out of the file. `ctranslate2` runs any Whisper because the language list
comes from the vocabulary. If your change needs a branch on a model's *name*, the design is wrong —
put the knowledge in the file format, the catalog, or a family spec.

**Nothing blocking runs on the event loop.** A llama.cpp decode step and a faster-whisper
`transcribe()` are blocking C calls. Run them in an executor. Iterating faster-whisper's lazy
segment generator on the loop froze every connection for ~200 ms per window before it was moved
into the worker thread.

**New runtimes must pass the conformance kit.** `fusion_runtime.testing.conformance` checks
cancellation, sample rates, capability reporting and error types:

```python
from fusion_runtime.testing.conformance import assert_conforms, check_runtime
assert_conforms(await check_runtime(MyRuntime(spec)))
```

Runtimes register through entry points, so a new engine can live in its own package:

```toml
[project.entry-points."fusion_runtime.runtimes"]
"tts.my_engine" = "my_package.tts:MyEngineRuntime"
```

**Secrets are named, never stored.** Config and agent files hold the *name* of an environment
variable (`api_key_env="GROQ_API_KEY"`), never a key. That's what makes `agent.py` safe to commit.

**Telemetry never leaks.** Transcript and reply content is off by default, secrets are redacted
always, and stack traces are never sent to a client. If you add an event, it carries what happened
and not what was said.

**Errors say what to do.** `frun models pull qwen2.5-7b-q4` beats "model not found". The failure
a user actually hits is part of the feature.

## Pull requests

Keep a pull request to one change. Say what broke or what was missing, not only what you did —
the reason a change exists is the part that's hard to recover later. If the change fixes something
that was silently wrong, say how you noticed, because the next person will hit it too.

Tests for behaviour you changed, and run the integration suite locally if you touched a runtime,
the engine, or the audio path.

## Security

Don't open a public issue for a security problem. Email the address on the
[GitHub profile](https://github.com/SamarthUrs18) instead, and give it a few days before
disclosing.
