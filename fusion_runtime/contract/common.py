"""Pieces every runtime shares: requests, cancellation, capabilities, health, errors.

A runtime is one loaded model (or one remote endpoint) that every session
shares. It receives requests, not sessions: anything per-conversation
(history, turn state, voice defaults) belongs to the engine and arrives
inside each request. That is what lets the engine queue, batch and cancel
work across many conversations on one GPU.
"""
import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Mapping, Optional, Tuple

Stage = Literal["stt", "llm", "tts", "turn"]
STAGES: Tuple[Stage, ...] = ("stt", "llm", "tts")  # model stages the resolver handles
RUNTIME_STAGES: Tuple[Stage, ...] = STAGES + ("turn",)  # everything the registry can load (turn detectors too)


# ---- cancellation -----------------------------------------------------------------

class CancelToken:
    """Cancellation shared between the engine and a runtime.

    Thread-safe, because runtimes often do their work on worker threads
    (a llama.cpp decode loop, an ONNX session). Runtimes check `cancelled`
    between steps, or register a callback to interrupt a blocking call.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []
        self.reason: Optional[str] = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self._event.set()
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Run `callback` on cancel, or right away if already cancelled."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return
        callback()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled(self.reason or "cancelled")

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until cancelled or `timeout` seconds pass. Returns whether it was cancelled.

        Lets runtimes sleep, back off or wait on I/O while still reacting to
        cancellation immediately, even when cancel() is called from another thread.
        """
        if self.cancelled:
            return True
        loop = asyncio.get_running_loop()
        woken = asyncio.Event()

        def wake() -> None:
            try:
                loop.call_soon_threadsafe(woken.set)
            except RuntimeError:  # loop already closed
                pass

        self.add_callback(wake)
        try:
            await asyncio.wait_for(woken.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return self.cancelled
        finally:
            with self._lock:
                if wake in self._callbacks:
                    self._callbacks.remove(wake)


# ---- requests and model specs -------------------------------------------------------

@dataclass(kw_only=True)
class Request:
    """Fields every request carries. Stage requests add their own inputs."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: Optional[str] = None  # for scheduling fairness and logs; runtimes keep no session state
    cancel: CancelToken = field(default_factory=CancelToken)
    deadline: Optional[float] = None  # time.monotonic() value after which the result is useless
    language: Optional[str] = None  # BCP-47-ish code ("en", "hi", "pt-br"); None = runtime default or auto

    def remaining_s(self) -> Optional[float]:
        return None if self.deadline is None else self.deadline - time.monotonic()


@dataclass(frozen=True)
class ModelSpec:
    """What the resolver hands a runtime: which model to load and how.

    `model` is already resolved (an absolute path, a model name on a server,
    or a URL). `family` names the pre/post-processing a runtime applies when
    the file format alone doesn't describe it (for example ONNX TTS).
    `options` carries runtime settings from config (device, context size,
    base_url, api_key_env, default voice, ...).
    """

    stage: Stage
    runtime: str
    model: str
    family: Optional[str] = None
    options: Mapping[str, Any] = field(default_factory=dict)


# ---- capabilities and health ---------------------------------------------------------

@dataclass(frozen=True)
class Capabilities:
    """What a loaded runtime can do. The engine adapts to these instead of
    knowing anything about the model."""

    streaming_input: bool = False  # STT: consumes audio incrementally
    streaming_output: bool = True  # LLM/TTS: yields partial results
    languages: Optional[Tuple[str, ...]] = None  # None = any / unknown
    sample_rate: Optional[int] = None  # Hz of audio the runtime takes (STT) or produces (TTS)
    voices: Tuple[str, ...] = ()
    max_batch: int = 1  # requests per model call; >1 lets the scheduler batch
    max_concurrency: int = 1  # requests in flight at once on this loaded model
    memory_bytes: int = 0  # estimate once loaded, for admission control and `frun doctor`
    tools: bool = False  # LLM: accepts tool definitions and emits tool calls
    decodes_on_demand: bool = True  # LLM: works only while the caller waits for the next chunk (in process).
    # False for remote servers, which generate ahead into a buffer: time spent waiting per chunk then
    # measures delivery, not model speed, so the engine doesn't report tokens per second for them.

    def __post_init__(self) -> None:
        if self.max_batch < 1 or self.max_concurrency < 1:
            raise ValueError("max_batch and max_concurrency must be at least 1")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.memory_bytes < 0:
            raise ValueError("memory_bytes can't be negative")

    def supports_language(self, language: Optional[str]) -> bool:
        if language is None or self.languages is None:
            return True
        base = language.lower().split("-")[0]
        return any(lang.lower() == language.lower() or lang.lower().split("-")[0] == base for lang in self.languages)


@dataclass(frozen=True)
class Health:
    status: Literal["ok", "degraded", "down"]
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---- errors -----------------------------------------------------------------------

class AdapterError(Exception):
    """Base for errors a runtime reports. `retryable` tells the engine whether
    trying again (same or another replica) can succeed."""

    retryable = False


class Cancelled(AdapterError):
    """The request's CancelToken fired. Not a failure."""


class InvalidRequest(AdapterError):
    """The request can't be served as given (empty input, unsupported language or voice)."""


class ModelNotFound(AdapterError):
    """The model files or remote model don't exist."""


class UnsupportedModel(AdapterError):
    """The runtime can't run this model (unknown family, architecture or format)."""


class AuthFailed(AdapterError):
    """A remote endpoint rejected the credentials."""


class RateLimited(AdapterError):
    retryable = True

    def __init__(self, message: str = "rate limited", retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class Overloaded(AdapterError):
    """No capacity right now; another replica or a later retry may succeed."""

    retryable = True


class RuntimeFailure(AdapterError):
    """The model or endpoint failed while serving the request."""


# ---- base runtime ---------------------------------------------------------------------

class ModelRuntime(ABC):
    """One loaded model, shared by every session. Lifecycle: construct → load → serve → close."""

    stage: Stage

    def __init__(self, spec: ModelSpec) -> None:
        if spec.stage != self.stage:
            raise ValueError(f"{type(self).__name__} is a {self.stage} runtime, got a {spec.stage} model spec")
        self.spec = spec

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Valid after load()."""

    @abstractmethod
    async def load(self) -> None:
        """Load weights / open connections. Must not block the event loop."""

    async def close(self) -> None:
        """Release the model. Must be safe to call more than once."""

    def health(self) -> Health:
        return Health("ok")
