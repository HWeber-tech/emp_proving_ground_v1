"""Heuristic compute allocator for cognitive subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CognitiveTask:
    """Task metadata tracked by the cognitive scheduler."""

    name: str
    information_gain: float
    compute_cost: float
    priority: int = 0
    last_run: datetime | None = None
    min_allocation: float = 0.0
    max_allocation: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CognitiveTaskDecision:
    """Allocation decision returned by the scheduler."""

    task: CognitiveTask
    allocated_compute: float
    score: float
    reason: str | None = None

    @property
    def selected(self) -> bool:
        return self.allocated_compute > 0


class CognitiveScheduler:
    """Allocate compute budget to tasks by information-gain density."""

    def allocate(
        self,
        tasks: Sequence[CognitiveTask],
        *,
        compute_budget: float,
    ) -> list[CognitiveTaskDecision]:
        if not tasks:
            return []

        if compute_budget <= 0:
            return [
                CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=0.0,
                    reason="no_budget_available",
                )
                for task in tasks
            ]

        decisions: dict[str, CognitiveTaskDecision] = {}
        candidates: list[tuple[float, CognitiveTask]] = []

        for task in tasks:
            if task.compute_cost <= 0:
                decisions[task.name] = CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=0.0,
                    reason="invalid_compute_cost",
                )
                continue
            if task.information_gain <= 0:
                decisions[task.name] = CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=0.0,
                    reason="no_information_gain",
                )
                continue
            score = task.information_gain / task.compute_cost
            candidates.append((score, task))

        candidates.sort(
            key=lambda pair: (pair[0], pair[1].priority, pair[1].information_gain),
            reverse=True,
        )

        remaining = compute_budget
        for score, task in candidates:
            if remaining <= 0:
                decisions[task.name] = CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=score,
                    reason="budget_exhausted",
                )
                continue

            allocation = min(task.compute_cost, remaining)
            minimum = max(task.min_allocation, 0.0)
            if allocation < minimum:
                decisions[task.name] = CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=score,
                    reason="insufficient_budget_for_minimum",
                )
                continue

            if task.max_allocation is not None:
                allocation = min(allocation, task.max_allocation)

            remaining -= allocation
            decisions[task.name] = CognitiveTaskDecision(
                task=task,
                allocated_compute=float(allocation),
                score=score,
                reason=None,
            )

        for task in tasks:
            if task.name not in decisions:
                decisions[task.name] = CognitiveTaskDecision(
                    task=task,
                    allocated_compute=0.0,
                    score=0.0,
                    reason="not_selected",
                )

        return sorted(
            decisions.values(),
            key=lambda decision: (
                0 if decision.selected else 1,
                -decision.score,
                -decision.task.priority,
                decision.task.name,
            ),
        )


__all__ = ["CognitiveScheduler", "CognitiveTask", "CognitiveTaskDecision"]
