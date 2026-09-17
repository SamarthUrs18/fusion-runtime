"""Sharing one loaded model between every conversation.

Each loaded model gets a ModelScheduler: a queue in front of it with a
concurrency limit. A request waits for a slot, holds it while the model works
(for a streamed reply, until the stream is closed) and releases it.

Who goes next is decided by deadline, not arrival: the waiting request whose
`deadline` is soonest gets the free slot; requests without a deadline go
after those with one, in arrival order. The deadline is read when a slot
frees, so the engine can move it while a request waits (for example as a
call's buffered audio runs down). This is where playback-aware scheduling
plugs in; today the engine sets deadlines from the turn's start.

A request is never dropped for missing its deadline; it just loses priority.
Admission control is the queue limit: past `max_queue` waiters a new request
fails fast with Overloaded instead of making everyone slower.

Everything here runs on the event loop; cancellation may come from any
thread through the request's CancelToken.
"""
import asyncio
import contextlib
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional, TypeVar

from fusion_runtime.contract.common import Cancelled, Overloaded, Request
from fusion_runtime.telemetry import telemetry

T = TypeVar("T")


@dataclass
class _Waiter:
    request: Request
    seq: int
    enqueued: float
    future: asyncio.Future = field(repr=False)

    def priority(self):
        deadline = self.request.deadline
        return (deadline if deadline is not None else math.inf, self.seq)


@dataclass
class SlotInfo:
    """What a request experienced getting its slot."""

    wait_ms: float
    queued_ahead: int  # requests already waiting when this one arrived


class ModelScheduler:
    """A concurrency-limited, deadline-ordered queue in front of one loaded model."""

    def __init__(self, stage: str, model: str = "", max_concurrency: int = 1, max_queue: Optional[int] = None):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_queue is not None and max_queue < 0:
            raise ValueError("max_queue can't be negative")
        self.stage = stage
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self._in_flight = 0
        self._waiters: List[_Waiter] = []
        self._seq = itertools.count()
        self.completed = 0
        self.rejected = 0

    # ---- state ------------------------------------------------------------------------

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queued(self) -> int:
        return len(self._waiters)

    def snapshot(self) -> Dict[str, object]:
        """Current load, for /health, /metrics and admission decisions."""
        now = time.monotonic()
        oldest = min((w.enqueued for w in self._waiters), default=None)
        return {
            "stage": self.stage, "model": self.model,
            "in_flight": self._in_flight, "queued": len(self._waiters),
            "max_concurrency": self.max_concurrency, "max_queue": self.max_queue,
            "oldest_wait_ms": round((now - oldest) * 1000, 1) if oldest is not None else 0.0,
            "completed": self.completed, "rejected": self.rejected,
        }

    # ---- slots ------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def slot(self, request: Request) -> AsyncIterator[SlotInfo]:
        """Hold one of the model's slots for the duration of the block."""
        info = await self._acquire(request)
        try:
            yield info
        finally:
            self._release()

    async def run(self, request: Request, work: Callable[[], Awaitable[T]]) -> T:
        """Run `work()` while holding a slot."""
        async with self.slot(request):
            return await work()

    async def _acquire(self, request: Request) -> SlotInfo:
        request.cancel.raise_if_cancelled()
        started = time.monotonic()
        ahead = len(self._waiters)
        if self._in_flight < self.max_concurrency and not self._waiters:
            self._in_flight += 1
            return self._granted(request, started, ahead)

        if self.max_queue is not None and len(self._waiters) >= self.max_queue:
            self.rejected += 1
            telemetry.emit("scheduler.rejected", level="warning", stage=self.stage, request_id=request.id,
                           session_id=request.session_id, model=self.model, queued=len(self._waiters),
                           max_queue=self.max_queue,
                           hint="the model is at capacity; add capacity or lower concurrent calls")
            raise Overloaded(f"{self.stage} model is at capacity ({len(self._waiters)} requests waiting)")

        loop = asyncio.get_running_loop()
        waiter = _Waiter(request, next(self._seq), started, loop.create_future())
        self._waiters.append(waiter)

        def on_cancel() -> None:
            def cancel_waiter() -> None:
                if not waiter.future.done():
                    waiter.future.set_exception(Cancelled(request.cancel.reason or "cancelled"))
            try:
                loop.call_soon_threadsafe(cancel_waiter)
            except RuntimeError:  # loop closed
                pass

        request.cancel.add_callback(on_cancel)
        try:
            await waiter.future  # resolved by _release() with the slot already counted for us
        except BaseException:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            elif waiter.future.done() and not waiter.future.cancelled() and waiter.future.exception() is None:
                self._release()  # granted just as we were cancelled: hand the slot on
            raise
        return self._granted(request, started, ahead)

    def _granted(self, request: Request, started: float, ahead: int) -> SlotInfo:
        info = SlotInfo(wait_ms=(time.monotonic() - started) * 1000, queued_ahead=ahead)
        telemetry.emit("scheduler.slot", level="debug", stage=self.stage, request_id=request.id,
                       session_id=request.session_id, duration_ms=info.wait_ms, model=self.model,
                       queued_ahead=ahead, in_flight=self._in_flight)
        return info

    def _release(self) -> None:
        self._in_flight -= 1
        self.completed += 1
        while self._waiters and self._in_flight < self.max_concurrency:
            waiter = min(self._waiters, key=_Waiter.priority)
            self._waiters.remove(waiter)
            if waiter.future.done():  # cancelled while waiting
                continue
            self._in_flight += 1
            waiter.future.set_result(None)
