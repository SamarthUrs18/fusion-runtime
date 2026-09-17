"""ModelScheduler: concurrency limits, deadline order, cancellation and admission control."""
import asyncio
import threading
import time

import pytest

from fusion_runtime.contract.common import Cancelled, Overloaded, Request
from fusion_runtime.engine.scheduler import ModelScheduler


async def hold(scheduler: ModelScheduler, request: Request, order: list, release: asyncio.Event):
    async with scheduler.slot(request):
        order.append(request.id)
        await release.wait()


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_requests_run_immediately_while_slots_are_free():
    scheduler = ModelScheduler("llm", max_concurrency=2)
    async with scheduler.slot(Request()) as first, scheduler.slot(Request()) as second:
        assert first.wait_ms < 50 and second.queued_ahead == 0
        assert scheduler.in_flight == 2
    assert scheduler.in_flight == 0 and scheduler.completed == 2


async def test_concurrency_limit_queues_the_rest():
    scheduler = ModelScheduler("llm", max_concurrency=1)
    release = asyncio.Event()
    order = []
    tasks = [asyncio.create_task(hold(scheduler, Request(id=str(i)), order, release)) for i in range(3)]
    await settle()
    assert order == ["0"] and scheduler.queued == 2
    release.set()
    await asyncio.gather(*tasks)
    assert order == ["0", "1", "2"]


async def test_soonest_deadline_goes_first_and_no_deadline_goes_last():
    scheduler = ModelScheduler("llm", max_concurrency=1)
    gate, release = asyncio.Event(), asyncio.Event()
    order = []
    blocker = asyncio.create_task(hold(scheduler, Request(id="busy"), order, gate))
    await settle()
    now = time.monotonic()
    waiting = [
        asyncio.create_task(hold(scheduler, Request(id="none"), order, release)),
        asyncio.create_task(hold(scheduler, Request(id="late", deadline=now + 5), order, release)),
        asyncio.create_task(hold(scheduler, Request(id="soon", deadline=now + 1), order, release)),
    ]
    await settle()
    gate.set()
    release.set()
    await asyncio.gather(blocker, *waiting)
    assert order == ["busy", "soon", "late", "none"]


async def test_deadline_changes_while_waiting_are_respected():
    scheduler = ModelScheduler("llm", max_concurrency=1)
    gate, release = asyncio.Event(), asyncio.Event()
    order = []
    blocker = asyncio.create_task(hold(scheduler, Request(id="busy"), order, gate))
    await settle()
    now = time.monotonic()
    a, b = Request(id="a", deadline=now + 1), Request(id="b", deadline=now + 2)
    tasks = [asyncio.create_task(hold(scheduler, r, order, release)) for r in (a, b)]
    await settle()
    b.deadline = now  # b's call is about to run out of audio
    gate.set()
    release.set()
    await asyncio.gather(blocker, *tasks)
    assert order == ["busy", "b", "a"]


async def test_cancel_token_from_another_thread_removes_a_waiter():
    scheduler = ModelScheduler("llm", max_concurrency=1)
    gate = asyncio.Event()
    order = []
    blocker = asyncio.create_task(hold(scheduler, Request(id="busy"), order, gate))
    await settle()
    waiting = Request(id="w")
    task = asyncio.create_task(hold(scheduler, waiting, order, gate))
    await settle()
    threading.Thread(target=waiting.cancel.cancel, args=("barge_in",)).start()
    with pytest.raises(Cancelled, match="barge_in"):
        await asyncio.wait_for(task, 1)
    assert scheduler.queued == 0
    gate.set()
    await blocker
    assert scheduler.in_flight == 0


async def test_already_cancelled_request_never_takes_a_slot():
    scheduler = ModelScheduler("llm")
    request = Request()
    request.cancel.cancel()
    with pytest.raises(Cancelled):
        async with scheduler.slot(request):
            pass
    assert scheduler.in_flight == 0


async def test_task_cancelled_while_waiting_or_just_granted_frees_everything():
    scheduler = ModelScheduler("llm", max_concurrency=1)
    gate, release = asyncio.Event(), asyncio.Event()
    order = []
    blocker = asyncio.create_task(hold(scheduler, Request(id="busy"), order, gate))
    await settle()
    waiting = asyncio.create_task(hold(scheduler, Request(id="w"), order, release))
    await settle()
    gate.set()
    await blocker  # the slot is now granted to "w", but its task hasn't resumed yet
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert scheduler.in_flight == 0 and scheduler.queued == 0
    async with scheduler.slot(Request()):  # still usable
        assert scheduler.in_flight == 1


async def test_errors_inside_the_slot_release_it():
    scheduler = ModelScheduler("llm")
    with pytest.raises(RuntimeError):
        async with scheduler.slot(Request()):
            raise RuntimeError("model failed")
    assert scheduler.in_flight == 0


async def test_full_queue_rejects_new_requests_fast():
    scheduler = ModelScheduler("llm", max_concurrency=1, max_queue=1)
    gate = asyncio.Event()
    order = []
    tasks = [asyncio.create_task(hold(scheduler, Request(id=str(i)), order, gate)) for i in range(2)]
    await settle()
    with pytest.raises(Overloaded):
        async with scheduler.slot(Request()):
            pass
    assert scheduler.rejected == 1
    snapshot = scheduler.snapshot()
    assert (snapshot["in_flight"], snapshot["queued"], snapshot["rejected"]) == (1, 1, 1)
    gate.set()
    await asyncio.gather(*tasks)


async def test_run_holds_a_slot_for_the_work():
    scheduler = ModelScheduler("stt", max_concurrency=1)

    async def work():
        assert scheduler.in_flight == 1
        return "text"

    assert await scheduler.run(Request(), work) == "text"


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        ModelScheduler("llm", max_concurrency=0)
    with pytest.raises(ValueError):
        ModelScheduler("llm", max_queue=-1)


async def test_wait_times_and_rejections_reach_prometheus():
    from fusion_runtime.telemetry import telemetry

    scheduler = ModelScheduler("tts", max_concurrency=1, max_queue=0)
    gate = asyncio.Event()
    task = asyncio.create_task(hold(scheduler, Request(), [], gate))
    await settle()
    with pytest.raises(Overloaded):
        async with scheduler.slot(Request()):
            pass
    gate.set()
    await task
    text = telemetry.metrics.render().decode()
    assert 'fusion_scheduler_wait_seconds_count{stage="tts"}' in text
    assert 'fusion_scheduler_rejected_total{stage="tts"}' in text
