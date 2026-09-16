"""Turning exceptions into errors people can act on: a stable code, a clean message, and a fix."""
import traceback
from typing import Optional, Tuple

from fusion_runtime.telemetry.events import ErrorInfo
from fusion_runtime.telemetry.redact import redact_secrets

STAGE_ATTR = "fusion_stage"  # set on an exception by the stage it escaped from


def tag_stage(exc: BaseException, stage: str) -> BaseException:
    """Remember which stage an exception came from (the innermost stage wins)."""
    if not getattr(exc, STAGE_ATTR, None):
        try:
            setattr(exc, STAGE_ATTR, stage)
        except Exception:  # some builtin exceptions don't take attributes
            pass
    return exc


def _classify(exc: BaseException) -> Tuple[str, bool, Optional[str]]:
    from fusion_runtime.contract import common as contract

    name = type(exc).__name__
    text = str(exc).lower()

    contract_codes = [
        (contract.Cancelled, "cancelled", False, None),
        (contract.ModelNotFound, "model_not_found", False, "Download it: frun models pull (check with: frun models list)"),
        (contract.UnsupportedModel, "unsupported_model", False,
         "This runtime can't run that model. Pick another runtime, or serve the model behind an OpenAI-compatible server"),
        (contract.AuthFailed, "auth_failed", False, "Check the API key environment variable for this endpoint, then run: frun doctor"),
        (contract.RateLimited, "rate_limited", True, "The provider is rate limiting; retry shortly or lower concurrency"),
        (contract.Overloaded, "overloaded", True, "No capacity right now; retry, or add capacity"),
        (contract.InvalidRequest, "invalid_request", False, None),
        (contract.RuntimeFailure, "runtime_failure", False, "The model runtime failed; see the stack trace and run: frun doctor"),
    ]
    for cls, code, retryable, fix in contract_codes:
        if isinstance(exc, cls):
            return code, retryable or bool(getattr(exc, "retryable", False)), fix

    if isinstance(exc, FileNotFoundError) and ("model" in text or "voices" in text):
        return "model_not_found", False, "Download it: frun models pull (check with: frun models list)"
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "missing_dependency", False, "Reinstall dependencies (pip install -e .) and run: frun doctor"
    if isinstance(exc, MemoryError) or "out of memory" in text or "cuda error: out of memory" in text:
        return "out_of_memory", False, "Use a smaller model, lower the context size, or run fewer sessions at once"
    # Errors from the OpenAI client (hosted LLMs, llama-server, vLLM), matched by name to avoid importing it
    if name == "AuthenticationError" or name == "PermissionDeniedError":
        return "auth_failed", False, "Check the API key environment variable for this endpoint, then run: frun doctor"
    if name == "RateLimitError":
        return "rate_limited", True, "The provider is rate limiting; retry shortly or lower concurrency"
    if name in ("APIConnectionError", "APITimeoutError") or isinstance(exc, (ConnectionError, TimeoutError)):
        return "endpoint_unreachable", True, "Check the endpoint URL and network, and that the server is running"
    if name == "NotFoundError" and "model" in text:
        return "model_not_found", False, "Check the model name on that endpoint"
    return "internal_error", False, "Run: frun doctor. If it persists, report it with the JSON log line for this error"


def describe_error(exc: BaseException, stage: Optional[str] = None, include_stack: bool = True) -> ErrorInfo:
    code, retryable, fix = _classify(exc)
    stack = None
    if include_stack and exc.__traceback__ is not None:
        stack = redact_secrets("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    message = redact_secrets(str(exc)) or type(exc).__name__
    return ErrorInfo(
        code=code,
        type=type(exc).__name__,
        message=message,
        retryable=retryable,
        fix=fix,
        stage=getattr(exc, STAGE_ATTR, None) or stage,
        stack=stack,
    )
