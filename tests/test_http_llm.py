"""OpenAI-compatible HTTP LLMs: the runtime, keys from the environment only, profiles, CLI and doctor."""
import asyncio
import json
import os

import httpx
import pytest
from fusion_runtime.catalog import entries_for_profile
from fusion_runtime.cli import _checks
from fusion_runtime.cli._checks import FAIL, INFO, OK, WARN
from fusion_runtime.cli.app import app
from fusion_runtime.config import (
    DEVELOPMENT_CONFIG,
    HYBRID_CONFIG,
    LLMConfig,
    load_profile,
    with_env_overrides,
)
from fusion_runtime.contract import (
    AuthFailed,
    InvalidRequest,
    LLMRequest,
    Message,
    ModelNotFound,
    ModelSpec,
    Overloaded,
    RateLimited,
    RuntimeFailure,
)
from fusion_runtime.resolver import resolve_stage_config
from fusion_runtime.runtimes.openai_http.llm import OpenAIHTTPLLM, probe_endpoint
from fusion_runtime.testing.conformance import assert_conforms, check_runtime
from typer.testing import CliRunner

URL = "http://llm.test/v1"
LLM_ENV = ("FUSION_LLM_URL", "FUSION_LLM_MODEL", "FUSION_LLM_API_KEY_ENV")


class FakeServer:
    """An OpenAI-compatible server in memory: /models and streaming /chat/completions."""

    def __init__(self, words=("We", " open", " at", " nine."), step_s=0.005, key=None, status=200, models=("m",)):
        self.words, self.step_s, self.key, self.status, self.models = list(words), step_s, key, status, models
        self.bodies = []
        self.auth_headers = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.append(request.headers.get("authorization"))
        if self.key and request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": {"message": "invalid api key"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        if self.status != 200:
            headers = {"retry-after": "2"} if self.status == 429 else {}
            return httpx.Response(self.status, json={"error": {"message": "nope"}}, headers=headers)
        body = json.loads(request.content)
        self.bodies.append(body)
        words = self.words[: body.get("max_tokens") or len(self.words)]
        step_s = self.step_s

        async def events():
            for word in words:
                await asyncio.sleep(step_s)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': word}, 'finish_reason': None}]})}\n\n".encode()
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events())

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)


def runtime(server: FakeServer, **options) -> OpenAIHTTPLLM:
    rt = OpenAIHTTPLLM(ModelSpec(stage="llm", runtime="openai_http", model=URL, options={"model_name": "m", **options}))
    rt.transport = server.transport
    return rt


def request(**kw):
    return LLMRequest(messages=[Message("system", "be brief"), Message("user", "hours?")], **kw)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for name in LLM_ENV + ("TEST_LLM_KEY",):
        monkeypatch.delenv(name, raising=False)


# ---- runtime -------------------------------------------------------------------------------

async def test_runtime_passes_the_conformance_kit():
    assert_conforms(await check_runtime(runtime(FakeServer(words=["w"] * 200, step_s=0.01))))


async def test_streams_text_and_sends_the_request_as_given():
    server = FakeServer()
    rt = runtime(server)
    await rt.load()
    chunks = [c async for c in rt.generate(request(max_tokens=3, stop=("\n",)))]
    assert "".join(c.text for c in chunks) == "We open at"
    assert chunks[-1].finish_reason == "stop"
    body = server.bodies[0]
    assert body["model"] == "m" and body["stream"] is True and body["stop"] == ["\n"]
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hours?"}]
    assert server.auth_headers[-1] is None  # no key configured: none sent
    assert rt.capabilities.decodes_on_demand is False
    await rt.close()


async def test_key_comes_from_the_named_environment_variable(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-test-123")
    server = FakeServer(key="sk-test-123")
    rt = runtime(server, api_key_env="TEST_LLM_KEY")
    await rt.load()
    assert "".join([c.text async for c in rt.generate(request())]) == "We open at nine."
    await rt.close()


async def test_missing_or_wrong_key_fails_at_load():
    with pytest.raises(AuthFailed, match="TEST_LLM_KEY"):
        await runtime(FakeServer(), api_key_env="TEST_LLM_KEY").load()


async def test_rejected_key_fails_at_load(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "wrong")
    with pytest.raises(AuthFailed, match="rejected the API key"):
        await runtime(FakeServer(key="right"), api_key_env="TEST_LLM_KEY").load()


async def test_model_name_is_required():
    with pytest.raises(InvalidRequest, match="model_name"):
        await OpenAIHTTPLLM(ModelSpec(stage="llm", runtime="openai_http", model=URL)).load()


async def test_unreachable_server_fails_at_load_with_a_hint():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    rt = runtime(FakeServer())
    rt.transport = httpx.MockTransport(refuse)
    with pytest.raises(RuntimeFailure, match="is the server running"):
        await rt.load()


@pytest.mark.parametrize("status, error", [(404, ModelNotFound), (429, RateLimited), (503, Overloaded),
                                           (400, InvalidRequest), (500, RuntimeFailure)])
async def test_http_errors_become_contract_errors(status, error):
    rt = runtime(FakeServer(status=status))
    await rt.load()
    with pytest.raises(error) as caught:
        async for _ in rt.generate(request()):
            pass
    if status == 429:
        assert caught.value.retry_after_s == 2
    await rt.close()


async def test_probe_reports_models_and_never_the_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-secret")
    report = await probe_endpoint(URL, "TEST_LLM_KEY", transport=FakeServer(key="sk-secret", models=("a", "b")).transport)
    assert report.reachable and report.auth_ok and report.models == ["a", "b"]
    assert report.lists("a") is True and report.lists("z") is False
    assert "sk-secret" not in repr(report)


# ---- config ----------------------------------------------------------------------------------

def test_keys_are_refused_in_config():
    with pytest.raises(ValueError, match="api_key_env"):
        LLMConfig(api_key="sk-live-123")


def test_hybrid_profile_reads_the_key_from_the_environment():
    resolved = resolve_stage_config("llm", HYBRID_CONFIG.llm)
    assert resolved.spec.runtime == "openai_http"
    assert resolved.spec.options["api_key_env"] == "OPENAI_API_KEY" and "api_key" not in resolved.spec.options
    assert "llm" not in {e.stage for e in entries_for_profile(HYBRID_CONFIG)}


def test_env_overrides_point_any_profile_at_an_endpoint():
    config = with_env_overrides(DEVELOPMENT_CONFIG, {"FUSION_LLM_URL": "http://localhost:8080/v1",
                                                     "FUSION_LLM_MODEL": "llama", "FUSION_LLM_API_KEY_ENV": "GROQ_KEY"})
    spec = resolve_stage_config("llm", config.llm).spec
    assert (spec.runtime, spec.model, spec.options["model_name"], spec.options["api_key_env"]) == (
        "openai_http", "http://localhost:8080/v1", "llama", "GROQ_KEY")
    assert DEVELOPMENT_CONFIG.llm.provider.value == "llama_cpp"  # the profile itself is untouched
    assert "llm" not in {e.stage for e in entries_for_profile(config)}


def test_env_url_without_a_model_name_is_explained():
    with pytest.raises(ValueError, match="FUSION_LLM_MODEL"):
        load_profile("development", {"FUSION_LLM_URL": "http://localhost:8080/v1"})
    with pytest.raises(ValueError, match="unknown profile"):
        load_profile("staging", {})


def test_config_can_name_a_runtime_and_any_model_reference():
    config = LLMConfig(runtime="openai_http", model="http://localhost:9000/v1", options={"model_name": "x"})
    spec = resolve_stage_config("llm", config).spec
    assert (spec.runtime, spec.options["model_name"]) == ("openai_http", "x")
    plugin = resolve_stage_config("llm", LLMConfig(runtime="my_pkg.llm:Engine", model="engine://local"))
    assert (plugin.source, plugin.spec.model) == ("plugin", "engine://local")


# ---- frun up / doctor -------------------------------------------------------------------------------

cli = CliRunner()


def test_up_refuses_a_missing_key_before_starting(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("server must not start"))
    monkeypatch.setattr("fusion_runtime.cli._checks.missing_models", lambda *a, **k: [])
    result = cli.invoke(app, ["up", "--config", "hybrid"])
    assert result.exit_code == 1 and "export OPENAI_API_KEY" in result.output


def test_up_llm_flags_set_the_endpoint_for_the_server(monkeypatch):
    for name in LLM_ENV:
        monkeypatch.setenv(name, "")  # restored after the test
    monkeypatch.delenv("FUSION_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: calls.append(k))
    monkeypatch.setattr("fusion_runtime.cli._checks.missing_models", lambda *a, **k: [])
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    result = cli.invoke(app, ["up", "--llm-url", "http://localhost:8080/v1", "--llm-model", "qwen-local"])
    assert result.exit_code == 0, result.output
    assert calls and os.environ["FUSION_LLM_URL"] == "http://localhost:8080/v1"
    assert "LLM: qwen-local at http://localhost:8080/v1" in result.output


def test_up_turn_flags_reach_the_server(monkeypatch):
    for name in LLM_ENV + ("FUSION_TURN_DETECTOR", "FUSION_TURN_WAIT_MS", "FUSION_INTERRUPT_AFTER_MS"):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("FUSION_CONFIG", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.setattr("fusion_runtime.cli._checks.missing_models", lambda *a, **k: [])
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    result = cli.invoke(app, ["up", "--turn-wait-ms", "1200", "--turn-detector", "my_pkg.turns:Model",
                              "--interrupt-after-ms", "450"])
    assert result.exit_code == 0, result.output
    assert os.environ["FUSION_TURN_WAIT_MS"] == "1200" and os.environ["FUSION_TURN_DETECTOR"] == "my_pkg.turns:Model"
    assert os.environ["FUSION_INTERRUPT_AFTER_MS"] == "450"
    assert "agent answers after 1200 ms of silence" in result.output and "talking over it for 450 ms interrupts" in result.output


def test_up_url_without_model_is_an_error(monkeypatch):
    monkeypatch.setenv("FUSION_LLM_URL", "")
    result = cli.invoke(app, ["up", "--llm-url", "http://localhost:8080/v1"])
    assert result.exit_code == 1 and "FUSION_LLM_MODEL" in result.output


def _fake_probe(monkeypatch, **report):
    from fusion_runtime.runtimes.openai_http import llm as http_llm

    async def probe(url, api_key_env=None, timeout_s=5.0, transport=None):
        return http_llm.EndpointReport(url=url, **report)

    monkeypatch.setattr(http_llm, "probe_endpoint", probe)


def test_doctor_without_hybrid_key_is_informational_and_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _fake_probe(monkeypatch, reachable=False)  # must not be consulted
    results = _checks.check_llm_endpoint()
    assert [r.status for r in results] == [INFO, INFO]


def test_doctor_checks_a_configured_endpoint(monkeypatch):
    monkeypatch.setenv("FUSION_LLM_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FUSION_LLM_MODEL", "qwen-local")
    _fake_probe(monkeypatch, reachable=True, auth_ok=True, models=["other"])
    statuses = {r.message: r.status for r in _checks.check_llm_endpoint()}
    assert statuses["http://localhost:8080/v1 is reachable"] == OK
    assert any(s == WARN and "isn't in the endpoint's model list" in m for m, s in statuses.items())


def test_doctor_fails_an_unreachable_configured_endpoint(monkeypatch):
    monkeypatch.setenv("FUSION_LLM_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("FUSION_LLM_MODEL", "qwen-local")
    _fake_probe(monkeypatch, reachable=False, error=RuntimeFailure("can't reach it; is the server running?"))
    assert _checks.check_llm_endpoint()[-1].status == FAIL
