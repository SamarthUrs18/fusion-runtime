"""Watches the event loop for blocking: the failure that silently freezes audio input,
barge-in detection and every other conversation on the server.

A stall is reported with `where`: the event loop thread's stack, caught by a watchdog
thread while the stall was still going on. Without it a stall says only that something
blocked; on the 4090 pod (19 Nov) finding the first cause took a separate experiment.
Only code locations are recorded, never values.
"""
import asyncio
import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

from fusion_runtime.telemetry.hub import Telemetry
from fusion_runtime.telemetry.hub import telemetry as default_hub


class LoopMonitor:
    def __init__(
        self,
        hub: Optional[Telemetry] = None,
        interval_s: float = 0.1,
        stall_ms: float = 100.0,
        window_s: float = 10.0,
        report_every_s: float = 10.0,
    ) -> None:
        self.hub = hub or default_hub
        self.interval_s = interval_s
        self.stall_ms = stall_ms
        self.window_s = window_s
        self.report_every_s = report_every_s
        self._lags: deque = deque()
        self._task: Optional[asyncio.Task] = None
        self._last_stall_report = 0.0
        self._loop_thread: Optional[int] = None
        self._beat = time.monotonic()  # when the loop last came round
        self._caught: Optional[str] = None  # the loop's stack, caught mid-stall
        self._stopped = threading.Event()
        self._watchdog: Optional[threading.Thread] = None

    @property
    def max_lag_ms(self) -> float:
        return max((lag for _, lag in self._lags), default=0.0)

    def start(self) -> None:
        if self._task is None:
            self._loop_thread = threading.get_ident()
            self._beat = time.monotonic()
            self._task = asyncio.get_running_loop().create_task(self._run())
            self._stopped.clear()
            self._watchdog = threading.Thread(target=self._watch, name="fusion-loop-watchdog", daemon=True)
            self._watchdog.start()

    def _watch(self) -> None:
        """Look at the loop thread while it's late: afterwards, the blocking code has gone."""
        check_s = max(self.stall_ms / 1000 / 5, 0.005)
        while not self._stopped.wait(check_s):
            late_ms = (time.monotonic() - self._beat - self.interval_s) * 1000
            if late_ms >= self.stall_ms / 2 and self._caught is None and self._loop_thread is not None:
                frame = sys._current_frames().get(self._loop_thread)
                if frame is not None:
                    self._caught = where(frame)

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        last_report = time.monotonic()
        while True:
            before = time.monotonic()
            await asyncio.sleep(self.interval_s)
            now = time.monotonic()
            self._beat = now
            caught, self._caught = self._caught, None
            lag_ms = max(0.0, (now - before - self.interval_s) * 1000)
            self._lags.append((now, lag_ms))
            while self._lags and self._lags[0][0] < now - self.window_s:
                self._lags.popleft()

            if lag_ms >= self.stall_ms and now - self._last_stall_report >= 1.0:
                self._last_stall_report = now
                self.hub.emit(
                    "event_loop.stall", level="warning", stage="server", duration_ms=lag_ms,
                    where=caught or None,
                    hint="Something ran blocking code on the event loop; audio and barge-in were frozen meanwhile",
                )
            if now - last_report >= self.report_every_s:
                last_report = now
                self.hub.emit("event_loop.lag", level="debug", stage="server", max_lag_ms=round(self.max_lag_ms, 1))


def where(frame, depth: int = 6) -> str:
    """The innermost frames of a stack, innermost first: "f (path:line) < g (path:line)", paths shortened."""
    parts = []
    for entry in reversed(traceback.extract_stack(frame)[-depth:]):
        path = entry.filename
        if "site-packages" + os.sep in path:
            path = path.split("site-packages" + os.sep, 1)[1]
        elif "fusion_runtime" + os.sep in path:
            path = "fusion_runtime/" + path.split("fusion_runtime" + os.sep, 1)[1]
        else:
            path = os.path.basename(path)
        parts.append(f"{entry.name} ({path}:{entry.lineno})")
    return " < ".join(parts)
