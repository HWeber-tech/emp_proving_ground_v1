from __future__ import annotations

import math

import pytest

import numpy as np

from src.thinking.learning.l2sp_rehearsal import (
    EquityRehearsalPlan,
    L2SPRegularizer,
    plan_equity_rehearsal,
)


def test_l2sp_penalty_numpy_mean() -> None:
    anchors = {
        "weights": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "bias": 0.5,
    }
    regularizer = L2SPRegularizer(anchors, strength=2.0, reduction="mean")

    current = {
        "weights": np.array([1.5, 1.5, 2.5], dtype=np.float32),
        "bias": 0.75,
    }

    penalty, breakdown = regularizer.penalty(current, return_breakdown=True)

    assert isinstance(penalty, float)
    assert penalty == pytest.approx(0.625)
    assert math.isclose(sum(breakdown.values()), penalty)
    assert breakdown["weights"] == pytest.approx(0.5)
    assert breakdown["bias"] == pytest.approx(0.125)


def test_l2sp_penalty_zero_strength() -> None:
    anchors = {"param": np.array([1.0, 2.0], dtype=np.float32)}
    regularizer = L2SPRegularizer(anchors, strength=0.0, reduction="sum")

    current = {"param": np.array([5.0, -3.0], dtype=np.float32)}
    penalty, breakdown = regularizer.penalty(current, return_breakdown=True)

    assert penalty == pytest.approx(0.0)
    assert breakdown["param"] == pytest.approx(0.0)


def test_l2sp_missing_parameter_raises() -> None:
    anchors = {"param": np.array([1.0], dtype=np.float32)}
    regularizer = L2SPRegularizer(anchors)

    with pytest.raises(KeyError):
        regularizer.penalty({})


def test_equity_rehearsal_plan_clamps_fraction() -> None:
    plan = plan_equity_rehearsal(128, target_fraction=0.1)
    assert isinstance(plan, EquityRehearsalPlan)
    assert 0.2 - 1e-9 <= plan.equity_fraction <= 0.3 + 1e-9
    assert plan.equity_batch_size + plan.fx_batch_size == 128


def test_equity_rehearsal_plan_small_batch_best_effort() -> None:
    plan = plan_equity_rehearsal(7, target_fraction=0.28)
    assert plan.equity_batch_size + plan.fx_batch_size == 7
    assert plan.equity_batch_size >= 1
    assert plan.equity_fraction == pytest.approx(plan.equity_batch_size / 7.0)


def test_l2sp_penalty_torch_sum() -> None:
    torch = pytest.importorskip("torch")

    anchors = {"layer": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    regularizer = L2SPRegularizer(anchors, strength=1.5, reduction="sum")

    current = {"layer": torch.tensor([1.5, 1.0], dtype=torch.float32, requires_grad=True)}
    penalty, breakdown = regularizer.penalty(current, return_breakdown=True)

    assert penalty.item() == pytest.approx(1.875)
    assert math.isclose(sum(breakdown.values()), penalty.item())

    penalty.backward()
    grad = current["layer"].grad
    assert grad is not None
    assert grad.tolist() == pytest.approx([1.5, -3.0])

