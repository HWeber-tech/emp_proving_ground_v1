from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.thinking.cognitive_scheduler import CognitiveScheduler, CognitiveTask


def _task(name: str, gain: float, cost: float, **kwargs: object) -> CognitiveTask:
    return CognitiveTask(name=name, information_gain=gain, compute_cost=cost, **kwargs)


def test_allocate_prioritises_information_density() -> None:
    scheduler = CognitiveScheduler()
    now = datetime.utcnow()
    tasks = [
        _task("high", 2.4, 1.0, priority=2, last_run=now - timedelta(minutes=30)),
        _task("medium", 1.2, 1.2, priority=1, last_run=now - timedelta(minutes=20)),
        _task("low", 0.6, 1.5, priority=0, last_run=now - timedelta(minutes=10)),
    ]

    decisions = scheduler.allocate(tasks, compute_budget=2.1)
    selected = [decision.task.name for decision in decisions if decision.selected]

    assert selected == ["high", "medium"]
    assert decisions[0].allocated_compute == pytest.approx(1.0)
    assert decisions[1].allocated_compute == pytest.approx(1.1, rel=1e-6)
    assert decisions[-1].reason in {"budget_exhausted", "insufficient_budget_for_minimum"}


def test_allocate_marks_invalid_and_zero_gain_tasks() -> None:
    scheduler = CognitiveScheduler()
    tasks = [
        _task("invalid_cost", 1.0, 0.0),
        _task("no_gain", 0.0, 1.0),
        _task("feasible", 0.9, 0.9, min_allocation=0.4),
    ]

    decisions = scheduler.allocate(tasks, compute_budget=0.5)
    decisions_by_name = {decision.task.name: decision for decision in decisions}

    assert decisions_by_name["invalid_cost"].reason == "invalid_compute_cost"
    assert decisions_by_name["no_gain"].reason == "no_information_gain"

    feasible = decisions_by_name["feasible"]
    assert feasible.selected is True
    assert feasible.reason is None
    assert feasible.allocated_compute == pytest.approx(0.5)
