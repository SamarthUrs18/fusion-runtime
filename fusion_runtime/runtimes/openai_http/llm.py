"""Chat models behind any OpenAI-compatible endpoint: vLLM, llama-server, Ollama, hosted APIs.

Speaks the protocol (POST /chat/completions with server-sent events) over
httpx, with no vendor SDK. The model reference is the base URL, for example
http://localhost:8000/v1 or https://api.groq.com/openai/v1.

API keys never live in config: `api_key_env` names the environment variable
that holds the key. Without it no Authorization header is sent, which is what
local servers expect.

Options:
    model_name    the model's name on the server (required)
    api_key_env   environment variable holding the API key
    timeout_s     connect/read timeout (default 30)
    verify        check the endpoint and key while loading (default true)
    max_concurrency  requests in flight at once (default 16)
"""
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

import httpx

from fusion_runtime.contract import (
    AdapterError,
    AuthFailed,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    LLMChunk,
    LLMRequest,
    LLMRuntime,
    ModelNotFound,
    Overloaded,
    RateLimited,
    RuntimeFailure,
)


@dataclass
class EndpointReport:
    """What a quick look at an endpoint found. Never contains the key."""

    url: str
    reachable: bool = False
    auth_ok: Optional[bool] = None  # None = couldn't tell
    models: List[str] = field(default_factory=list)  # empty when the server doesn't list models
    error: Optional[AdapterError] = None

    def lists(self, model_name: str) -> Optional[bool]:
        return None if not self.models else model_name in self.models


def api_key(api_key_env: Optional[str]) -> Optional[str]:
    return os.getenv(api_key_env) if api_key_env else None


async def probe_endpoint(url: str, api_key_env: Optional[str] = None, timeout_s: float = 5.0,
                         transport: Optional[httpx.AsyncBaseTransport] = None) -> EndpointReport:
    """GET {url}/models: is the server up, does the key work, which models does it serve."""
    report = EndpointReport(url=url)
    if api_key_env and not api_key(api_key_env):
        report.error = AuthFailed(f"the environment variable {api_key_env} isn't set")
        report.auth_ok = False
    async with _client(url, api_key_env, timeout_s, transport) as client:
        try:
            response = await client.get("/models")
        except httpx.HTTPError as e:
            report.error = report.error or _connection_error(url, e)
            return report
        report.reachable = True
        if response.status_code in (401, 403):
            report.auth_ok = False
            report.error = _status_error(response, url)
            return report
        if response.is_success:
            report.auth_ok = report.auth_ok if report.auth_ok is False else True
            try:
                report.models = [m["id"] for m in response.json().get("data", []) if "id" in m]
            except (ValueError, TypeError, AttributeError):
                pass
        return report


class OpenAIHTTPLLM(LLMRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.client: Optional[httpx.AsyncClient] = None
        self.transport: Optional[httpx.AsyncBaseTransport] = None  # tests plug in a fake server here
        self._health = Health("down", "not loaded")

    @property
    def model_name(self) -> Optional[str]:
        return self.spec.options.get("model_name")

    @property
    def api_key_env(self) -> Optional[str]:
        return self.spec.options.get("api_key_env")

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(max_concurrency=self.spec.options.get("max_concurrency", 16), decodes_on_demand=False)

    def health(self) -> Health:
        return self._health

    async def load(self) -> None:
        from fusion_runtime.telemetry import telemetry

        if not self.model_name:
            raise InvalidRequest(f"set the model name to use on {self.spec.model} (model_name)")
        if self.api_key_env and not api_key(self.api_key_env):
            raise AuthFailed(f"the API key for {self.spec.model} is missing: set the environment variable "
                             f"{self.api_key_env}")
        timeout = self.spec.options.get("timeout_s", 30.0)
        if self.spec.options.get("verify", True):
            report = await probe_endpoint(self.spec.model, self.api_key_env, min(timeout, 10.0), self.transport)
            if report.error is not None:
                raise report.error
            if report.lists(self.model_name) is False:
                # Some servers accept any name (llama-server) or list aliases; don't refuse, but say so
                telemetry.emit("llm.endpoint_model_unlisted", level="warning", stage="llm", url=self.spec.model,
                               model=self.model_name, listed=report.models[:10],
                               hint="check model_name if requests fail with model_not_found")
        self.client = _client(self.spec.model, self.api_key_env, timeout, self.transport)
        self._health = Health("ok")

    async def close(self) -> None:
        client, self.client = self.client, None
        self._health = Health("down", "closed")
        if client is not None:
            await client.aclose()

    async def generate(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        if not request.messages:
            raise InvalidRequest("messages must not be empty")
        if request.tools:
            raise InvalidRequest("tool calling isn't supported by the openai_http runtime yet")
        request.cancel.raise_if_cancelled()
        if self.client is None:
            raise RuntimeFailure("runtime not loaded")

        body: Dict = {
            "model": self.model_name,
            "messages": [_message(m) for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.stop:
            body["stop"] = list(request.stop)

        loop = asyncio.get_running_loop()
        try:
            async with self.client.stream("POST", "/chat/completions", json=body) as response:
                def close_on_cancel() -> None:
                    # Wakes a read that's waiting on a slow server; the loop below then raises Cancelled
                    try:
                        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(response.aclose()))
                    except RuntimeError:
                        pass

                request.cancel.add_callback(close_on_cancel)
                if not response.is_success:
                    await response.aread()
                    raise _status_error(response, self.spec.model)
                produced = 0
                async for line in response.aiter_lines():
                    if request.cancel.cancelled:
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        raise RuntimeFailure(f"{self.spec.model} sent a malformed stream event") from None
                    if event.get("error"):
                        raise RuntimeFailure(f"{self.spec.model} reported an error: {event['error']}")
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        produced += 1
                    if choice.get("finish_reason"):
                        yield LLMChunk(text=text, finish_reason=choice["finish_reason"],
                                       usage={"completion_tokens": produced})
                        return
                    if text:
                        yield LLMChunk(text=text)
                if request.cancel.cancelled:
                    raise Cancelled(request.cancel.reason or "cancelled")
                yield LLMChunk(finish_reason="stop", usage={"completion_tokens": produced})
        except AdapterError:
            raise
        except (httpx.HTTPError, httpx.StreamError) as e:
            if request.cancel.cancelled:
                raise Cancelled(request.cancel.reason or "cancelled") from None
            raise _connection_error(self.spec.model, e) from e


def _client(url: str, api_key_env: Optional[str], timeout_s: float,
            transport: Optional[httpx.AsyncBaseTransport]) -> httpx.AsyncClient:
    headers = {}
    key = api_key(api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return httpx.AsyncClient(base_url=url.rstrip("/"), headers=headers, transport=transport,
                             timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 5.0)))


def _message(message) -> Dict[str, str]:
    out = {"role": message.role, "content": message.content}
    if message.name:
        out["name"] = message.name
    if message.tool_call_id:
        out["tool_call_id"] = message.tool_call_id
    return out


def _status_error(response: httpx.Response, url: str) -> AdapterError:
    try:
        detail = response.json().get("error") or response.text
        if isinstance(detail, dict):
            detail = detail.get("message") or detail
    except (ValueError, AttributeError):
        detail = response.text
    detail = str(detail)[:300]
    status = response.status_code
    if status in (401, 403):
        return AuthFailed(f"{url} rejected the API key ({status}): {detail}")
    if status == 404:
        return ModelNotFound(f"{url} returned 404 (wrong URL or model name?): {detail}")
    if status == 429:
        retry_after = response.headers.get("retry-after")
        try:
            seconds = float(retry_after) if retry_after else None
        except ValueError:
            seconds = None
        return RateLimited(f"{url} is rate limiting requests: {detail}", retry_after_s=seconds)
    if status in (400, 413, 422):
        return InvalidRequest(f"{url} refused the request ({status}): {detail}")
    if status in (502, 503, 504):
        return Overloaded(f"{url} is unavailable or overloaded ({status}): {detail}")
    return RuntimeFailure(f"{url} failed ({status}): {detail}")


def _connection_error(url: str, error: Exception) -> RuntimeFailure:
    if isinstance(error, httpx.TimeoutException):
        return RuntimeFailure(f"{url} timed out ({type(error).__name__}); is the server running and responsive?")
    return RuntimeFailure(f"can't reach {url} ({type(error).__name__}: {error}); is the server running?")
