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

A model that takes several requests in one call (`batch()`, for speech to text)
gets them as a batch: requests that arrive while every slot is busy wait
together, and the next free slot takes up to `max_batch` of them, most urgent
first. There is no window to wait for company: a request that finds a slot free
runs at once, alone, so one caller is exactly as fast as without batching, and
batches form only when callers would otherwise queue.

Everything here runs on the event loop; cancellation may come from any
thread through the request's CancelToken.
"""
import asyncio
import contextlib
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar

from fusion_runtime.contract.common import Cancelled, Overloaded, Request
from fusion_runtime.telemetry import telemetry

T = TypeVar("T")
R = TypeVar("R")
BatchCall = Callable[[Sequence[Request]], Awaitable[List]]


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
        # Requests waiting to ride in the model's next batch call (batch())
        self._riders: List[_Waiter] = []
        self._batch_call: Optional[BatchCall] = None
        self._max_batch = 1
        self._batch_tasks: set = set()
        self.batches = 0

    # ---- state ------------------------------------------------------------------------

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queued(self) -> int:
        return len(self._waiters) + len(self._riders)

    def snapshot(self) -> Dict[str, object]:
        """Current load, for /health, /metrics and admission decisions."""
        now = time.monotonic()
        oldest = min((w.enqueued for w in self._waiters + self._riders), default=None)
        return {
            "stage": self.stage, "model": self.model,
            "in_flight": self._in_flight, "queued": self.queued, "max_batch": self._max_batch,
            "batches": self.batches,
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
        self._hand_on()

    def _hand_on(self) -> None:
        """A slot came free: give it to the most urgent slot waiter, else to the next batch."""
        while self._waiters and self._in_flight < self.max_concurrency:
            waiter = min(self._waiters, key=_Waiter.priority)
            self._waiters.remove(waiter)
            if waiter.future.done():  # cancelled while waiting
                continue
            self._in_flight += 1
            waiter.future.set_result(None)
        self._dispatch()

    # ---- batches ----------------------------------------------------------------------

    async def batch(self, request: Request, call: BatchCall, max_batch: int = 1):
        """Run `request` through `call(requests) -> results`, sharing the call with others.

        `call` takes a list of requests and returns one result per request, in order; it
        must be the same function for every request on this model. With `max_batch` 1
        this is a slot and a call of one.
        """
        if max_batch <= 1:
            async with self.slot(request):
                return (await call([request]))[0]
        request.cancel.raise_if_cancelled()
        self._batch_call, self._max_batch = call, max_batch
        if self.max_queue is not None and self.queued >= self.max_queue:
            self.rejected += 1
            telemetry.emit("scheduler.rejected", level="warning", stage=self.stage, request_id=request.id,
                           session_id=request.session_id, model=self.model, queued=self.queued,
                           max_queue=self.max_queue,
                           hint="the model is at capacity; add capacity or lower concurrent calls")
            raise Overloaded(f"{self.stage} model is at capacity ({self.queued} requests waiting)")

        loop = asyncio.get_running_loop()
        rider = _Waiter(request, next(self._seq), time.monotonic(), loop.create_future())
        self._riders.append(rider)

        def on_cancel() -> None:
            def cancel_rider() -> None:
                if rider in self._riders and not rider.future.done():
                    self._riders.remove(rider)
                    rider.future.set_exception(Cancelled(request.cancel.reason or "cancelled"))
            try:
                loop.call_soon_threadsafe(cancel_rider)
            except RuntimeError:  # loop closed
                pass

        request.cancel.add_callback(on_cancel)
        self._dispatch()
        try:
            return await rider.future
        finally:
            if rider in self._riders:  # the caller gave up while it waited
                self._riders.remove(rider)

    def _dispatch(self) -> None:
        """Start batches while slots are free; slot waiters go first (_hand_on)."""
        while self._riders and not self._waiters and self._in_flight < self.max_concurrency:
            self._riders.sort(key=_Waiter.priority)
            batch, self._riders = self._riders[:self._max_batch], self._riders[self._max_batch:]
            self._in_flight += 1
            task = asyncio.get_running_loop().create_task(self._run_batch(batch, self._batch_call))
            self._batch_tasks.add(task)
            task.add_done_callback(self._batch_tasks.discard)

    async def _run_batch(self, batch: List[_Waiter], call: BatchCall) -> None:
        started = time.monotonic()
        waits = [(started - w.enqueued) * 1000 for w in batch]
        for rider, wait_ms in zip(batch, waits, strict=True):
            telemetry.emit("scheduler.slot", level="debug", stage=self.stage, request_id=rider.request.id,
                           session_id=rider.request.session_id, duration_ms=wait_ms, model=self.model,
                           batch_size=len(batch), in_flight=self._in_flight)
        try:
            results = await call([w.request for w in batch])
            if len(results) != len(batch):
                raise RuntimeError(f"{self.stage} model returned {len(results)} results for {len(batch)} requests")
            for rider, result in zip(batch, results, strict=True):
                if not rider.future.done():
                    rider.future.set_result(result)
        except Exception as e:
            for rider in batch:
                if not rider.future.done():
                    rider.future.set_exception(e)
        except BaseException:  # the server is shutting down: nobody gets a result
            for rider in batch:
                rider.future.cancel()
            raise
        finally:
            self.batches += 1
            telemetry.emit("scheduler.batch", level="debug", stage=self.stage, model=self.model,
                           size=len(batch), max_batch=self._max_batch, queued=len(self._riders),
                           duration_ms=(time.monotonic() - started) * 1000,
                           oldest_wait_ms=round(max(waits), 1))
            self._in_flight -= 1
            self.completed += len(batch)
            self._hand_on()
