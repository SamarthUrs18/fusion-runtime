"""Chat models behind any OpenAI-compatible endpoint: vLLM, llama-server, Ollama, hosted APIs.

The model reference is the base URL; the model name on the server is the
`model_name` option. The API key is read from the environment variable named
by `api_key_env` (default OPENAI_API_KEY) and never stored in config.

Options: model_name, api_key_env, timeout_s.
"""
import os
from typing import AsyncIterator

from fusion_runtime.contract import (
    AuthFailed,
    Cancelled,
    Capabilities,
    Health,
    InvalidRequest,
    LLMChunk,
    LLMRequest,
    LLMRuntime,
    ModelNotFound,
    RateLimited,
    RuntimeFailure,
)


class OpenAIHTTPLLM(LLMRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.client = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(max_concurrency=self.spec.options.get("max_concurrency", 16))

    def health(self) -> Health:
        return Health("ok") if self.client is not None else Health("down", "client not created")

    async def load(self) -> None:
        import openai

        key_env = self.spec.options.get("api_key_env", "OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv(key_env) or "not-needed",  # local servers usually accept any key
            base_url=self.spec.model,
            timeout=self.spec.options.get("timeout_s", 30.0),
        )

    async def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            await client.close()

    async def generate(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        if not request.messages:
            raise InvalidRequest("messages must not be empty")
        request.cancel.raise_if_cancelled()
        if self.client is None:
            raise RuntimeFailure("client not created")
        model_name = self.spec.options.get("model_name")
        if not model_name:
            raise InvalidRequest("set the model name for this endpoint (model_name)")
        try:
            stream = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=list(request.stop) or None,
                stream=True,
            )
            produced = 0
            try:
                async for chunk in stream:
                    if request.cancel.cancelled:
                        raise Cancelled(request.cancel.reason or "cancelled")
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    text = choice.delta.content or ""
                    if text:
                        produced += 1
                    if choice.finish_reason:
                        yield LLMChunk(text=text, finish_reason=choice.finish_reason,
                                       usage={"completion_tokens": produced})
                        return
                    if text:
                        yield LLMChunk(text=text)
                yield LLMChunk(finish_reason="stop", usage={"completion_tokens": produced})
            finally:
                await stream.close()
        except (Cancelled, InvalidRequest):
            raise
        except Exception as e:
            raise _map_error(e) from e


def _map_error(e: Exception) -> Exception:
    name = type(e).__name__
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return AuthFailed(str(e))
    if name == "RateLimitError":
        return RateLimited(str(e))
    if name == "NotFoundError":
        return ModelNotFound(str(e))
    if name == "BadRequestError":
        return InvalidRequest(str(e))
    return RuntimeFailure(f"{name}: {e}")
