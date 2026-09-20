"""Watches the event loop for blocking: the failure that silently freezes audio input,
barge-in detection and every other conversation on the server."""
import asyncio
import time
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

    @property
    def max_lag_ms(self) -> float:
        return max((lag for _, lag in self._lags), default=0.0)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
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
            lag_ms = max(0.0, (now - before - self.interval_s) * 1000)
            self._lags.append((now, lag_ms))
            while self._lags and self._lags[0][0] < now - self.window_s:
                self._lags.popleft()

            if lag_ms >= self.stall_ms and now - self._last_stall_report >= 1.0:
                self._last_stall_report = now
                self.hub.emit(
                    "event_loop.stall", level="warning", stage="server", duration_ms=lag_ms,
                    hint="Something ran blocking code on the event loop; audio and barge-in were frozen meanwhile",
                )
            if now - last_report >= self.report_every_s:
                last_report = now
                self.hub.emit("event_loop.lag", level="debug", stage="server", max_lag_ms=round(self.max_lag_ms, 1))
