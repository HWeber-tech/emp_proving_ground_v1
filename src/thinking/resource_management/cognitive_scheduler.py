"""Cognitive Scheduler: allocate compute by information gain potential.

This module fulfils the roadmap item **Cognitive Scheduler** under the
"Cognitive Resource Management" initiative.  It provides a lightweight, fully
in-memory scheduler that decides which cognitive workloads should receive
compute budget based on their expected information gain, adjusted for compute
cost and fairness.  Idle workloads receive a configurable staleness boost so no
signal is permanently starved even when budget is tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import math
from typing import Callable, Mapping, MutableMapping

__all__ = [
    "CognitiveTask",
    "CognitiveSchedulerConfig",
    "TaskAllocation",
    "SchedulerCycle",
    "CognitiveScheduler",
]


_EARLIEST = datetime.min.replace(tzinfo=UTC)


def _ensure_aware(moment: datetime) -> datetime:
    """Coerce naive datetimes to UTC for internal storage."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


@dataclass(slots=True)
class CognitiveTask:
    """Describe a cognitive workload and its information gain characteristics."""

    task_id: str
    info_gain: float
    cost: float = 1.0
    min_interval: timedelta | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        task_id = self.task_id.strip()
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if not math.isfinite(self.info_gain):
            raise ValueError("info_gain must be finite")
        if self.cost <= 0 or not math.isfinite(self.cost):
            raise ValueError("cost must be a positive finite number")
        if self.min_interval is not None and self.min_interval < timedelta(0):
            raise ValueError("min_interval cannot be negative")
        object.__setattr__(self, "task_id", task_id)


@dataclass(slots=True)
class CognitiveSchedulerConfig:
    """Runtime configuration for :class:`CognitiveScheduler`."""

    fairness_weight: float = 0.5
    staleness_halflife: float = 600.0
    initial_boost: float = 1.5
    min_score: float = 0.0
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.fairness_weight < 0:
            raise ValueError("fairness_weight must be non-negative")
        if self.staleness_halflife < 0:
            raise ValueError("staleness_halflife cannot be negative")
        if self.initial_boost < 0:
            raise ValueError("initial_boost must be non-negative")
        if self.min_score < 0:
            raise ValueError("min_score must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(slots=True)
class TaskAllocation:
    """Allocation decision for a single cognitive workload."""

    task_id: str
    compute: float
    expected_info_gain: float
    score: float
    metadata: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "compute": self.compute,
            "expected_info_gain": self.expected_info_gain,
            "score": self.score,
            "metadata": dict(self.metadata) if isinstance(self.metadata, Mapping) else self.metadata,
        }


@dataclass(slots=True)
class SchedulerCycle:
    """Summary of a single scheduling cycle."""

    timestamp: datetime
    budget: float
    allocations: tuple[TaskAllocation, ...]
    unused_budget: float

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "budget": self.budget,
            "unused_budget": self.unused_budget,
            "allocations": [alloc.as_dict() for alloc in self.allocations],
        }


@dataclass(slots=True)
class _TaskState:
    task: CognitiveTask
    last_run: datetime | None = None
    total_allocated: float = 0.0
    total_expected_gain: float = 0.0
    last_score: float = 0.0


class CognitiveScheduler:
    """Allocate compute budget to cognitive workloads based on information gain."""

    def __init__(
        self,
        config: CognitiveSchedulerConfig | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config or CognitiveSchedulerConfig()
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._tasks: MutableMapping[str, _TaskState] = {}

    def register_task(self, task: CognitiveTask) -> None:
        """Add or update a cognitive workload."""

        state = self._tasks.get(task.task_id)
        if state is None:
            self._tasks[task.task_id] = _TaskState(task=task)
        else:
            state.task = task

    def remove_task(self, task_id: str) -> bool:
        """Remove a workload from the scheduler."""

        return self._tasks.pop(task_id, None) is not None

    def get_task(self, task_id: str) -> CognitiveTask | None:
        """Return the current task definition if registered."""

        state = self._tasks.get(task_id)
        return state.task if state else None

    def tasks(self) -> Mapping[str, CognitiveTask]:
        """Expose a snapshot of registered workloads."""

        return {task_id: state.task for task_id, state in self._tasks.items()}

    def schedule(
        self,
        budget: float,
        *,
        now: datetime | None = None,
        max_allocations: int | None = None,
    ) -> SchedulerCycle:
        """Allocate compute budget for the current cycle."""

        if budget < 0:
            raise ValueError("budget cannot be negative")

        timestamp = _ensure_aware(now or self._now())
        if budget == 0 or not self._tasks:
            return SchedulerCycle(timestamp=timestamp, budget=float(budget), allocations=(), unused_budget=float(budget))

        eligible: list[tuple[_TaskState, float]] = []
        for state in self._tasks.values():
            task = state.task
            # Reset score until recomputed
            state.last_score = 0.0

            if task.info_gain <= 0:
                continue
            if task.min_interval is not None and state.last_run is not None:
                delta = timestamp - state.last_run
                if delta < task.min_interval:
                    continue

            base_score = task.info_gain / max(task.cost, self._config.epsilon)
            staleness = self._compute_staleness_boost(state, timestamp)
            score = base_score * (1.0 + self._config.fairness_weight * staleness)
            if score < self._config.min_score:
                continue
            state.last_score = score
            eligible.append((state, score))

        if not eligible:
            return SchedulerCycle(timestamp=timestamp, budget=float(budget), allocations=(), unused_budget=float(budget))

        eligible.sort(key=self._sort_key)

        remaining = float(budget)
        allocations: list[TaskAllocation] = []
        limit = max_allocations if max_allocations is not None else len(eligible)

        for state, score in eligible[:limit]:
            if remaining <= self._config.epsilon:
                break

            task = state.task
            compute = min(task.cost, remaining)
            if compute <= self._config.epsilon:
                continue

            expected_gain = task.info_gain * (compute / task.cost)
            allocation = TaskAllocation(
                task_id=task.task_id,
                compute=compute,
                expected_info_gain=expected_gain,
                score=score,
                metadata=task.metadata,
            )
            allocations.append(allocation)
            remaining -= compute
            state.last_run = timestamp
            state.total_allocated += compute
            state.total_expected_gain += expected_gain

        unused_budget = max(0.0, remaining)
        return SchedulerCycle(
            timestamp=timestamp,
            budget=float(budget),
            allocations=tuple(allocations),
            unused_budget=unused_budget,
        )

    def _compute_staleness_boost(self, state: _TaskState, now: datetime) -> float:
        if state.last_run is None:
            return self._config.initial_boost

        if self._config.staleness_halflife == 0:
            return 0.0

        staleness_seconds = max(0.0, (now - state.last_run).total_seconds())
        return staleness_seconds / self._config.staleness_halflife

    @staticmethod
    def _sort_key(entry: tuple[_TaskState, float]) -> tuple[float, datetime, str]:
        state, score = entry
        last_run = state.last_run or _EARLIEST
        return (-score, last_run, state.task.task_id)

    def snapshot(self) -> dict[str, object]:
        """Return a diagnostic snapshot of scheduler state."""

        tasks: list[dict[str, object]] = []
        for state in self._tasks.values():
            tasks.append(
                {
                    "task_id": state.task.task_id,
                    "info_gain": state.task.info_gain,
                    "cost": state.task.cost,
                    "min_interval": state.task.min_interval.total_seconds()
                    if state.task.min_interval is not None
                    else None,
                    "last_run": state.last_run.isoformat() if state.last_run else None,
                    "total_allocated": state.total_allocated,
                    "total_expected_gain": state.total_expected_gain,
                    "last_score": state.last_score,
                }
            )
        return {
            "tasks": tasks,
            "task_count": len(tasks),
        }
