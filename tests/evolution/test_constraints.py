from __future__ import annotations

import pytest

from src.evolution.optimization.constraints import (
    ConstraintSet,
    HardConstraint,
    SoftConstraint,
)


def test_hard_constraint_blocks_candidate_and_tracks_violation() -> None:
    constraint = HardConstraint(
        name="max_drawdown",
        metric="max_dd",
        operator="<=",
        limit=0.1,
    )
    constraint_set = ConstraintSet(hard=(constraint,))

    result = constraint_set.evaluate({"max_dd": 0.15})

    assert result.satisfied is False
    assert result.hard_violations
    violation = result.hard_violations[0]
    assert violation.constraint == "max_drawdown"
    assert violation.details["value"] == pytest.approx(0.15)
    assert constraint_set.violation_counts["max_drawdown"] == 1
    assert constraint_set.violation_history[-1].kind == "hard"


def test_soft_constraint_accumulates_penalty_on_violation() -> None:
    constraint = SoftConstraint(
        name="target_sharpe",
        metric="sharpe",
        operator=">=",
        limit=1.8,
        weight=5.0,
    )
    constraint_set = ConstraintSet(soft=(constraint,))

    result = constraint_set.evaluate({"sharpe": 1.6})

    assert result.satisfied is True
    assert result.penalty == pytest.approx(5.0 * 0.2)
    assert result.soft_violations
    violation = result.soft_violations[0]
    assert violation.constraint == "target_sharpe"
    assert violation.penalty == pytest.approx(result.penalty)
    assert constraint_set.violation_counts["target_sharpe"] == 1


def test_constraints_pass_when_metrics_within_limits() -> None:
    hard_constraint = HardConstraint(
        name="max_risk",
        metric="variance",
        operator="<=",
        limit=0.05,
    )
    soft_constraint = SoftConstraint(
        name="return_target",
        metric="expected_return",
        operator=">=",
        limit=0.12,
        weight=1.5,
    )
    constraint_set = ConstraintSet(hard=(hard_constraint,), soft=(soft_constraint,))

    result = constraint_set.evaluate({"variance": 0.03, "expected_return": 0.15})

    assert result.satisfied is True
    assert result.penalty == pytest.approx(0.0)
    assert not result.violations
    assert constraint_set.violation_history == ()
