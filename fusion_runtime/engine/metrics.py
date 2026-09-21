"""Per-turn latency budgets and pipeline metrics."""
import time
from dataclasses import dataclass, field

from fusion_runtime.telemetry import telemetry


@dataclass
class LatencyBudget:
    """Tracks latency budget across pipeline stages."""
    total_ms: int
    spent_ms: float = 0
    stage_budgets: dict = field(default_factory=dict)

    def allocate(self, stage: str, ms: int) -> "StageBudget":
        self.stage_budgets[stage] = ms
        return StageBudget(self, stage, ms)

    def record(self, stage: str, ms: float):
        self.spent_ms += ms
        if stage in self.stage_budgets:
            over = ms - self.stage_budgets[stage]
            if over > 0:
                # Never print from library code: a deployment running --log-format json
                # would get one unstructured line in the middle of its log stream, and
                # stdout is the server's audio-adjacent hot path, not a debug channel.
                telemetry.emit("stage.over_budget", level="warning", stage=stage,
                               over_ms=round(over), budget_ms=self.stage_budgets[stage])


@dataclass
class StageBudget:
    """Context manager for stage latency tracking."""
    budget: LatencyBudget
    stage: str
    allocated_ms: int
    start_time: float = field(default_factory=time.perf_counter)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.budget.record(self.stage, elapsed)

    @property
    def remaining_ms(self) -> float:
        elapsed = (time.perf_counter() - self.start_time) * 1000
        return max(0, self.allocated_ms - elapsed)

    @property
    def is_exceeded(self) -> bool:
        return self.remaining_ms <= 0


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    stt_latency_ms: float = 0
    llm_first_token_ms: float = 0
    llm_total_ms: float = 0
    tts_first_chunk_ms: float = 0
    tts_total_ms: float = 0
    e2e_latency_ms: float = 0
    pipeline_start: float = field(default_factory=time.perf_counter)
    timestamp: float = field(default_factory=time.time)
