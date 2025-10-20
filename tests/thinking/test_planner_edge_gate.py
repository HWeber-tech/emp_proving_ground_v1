from __future__ import annotations

import pytest

from src.thinking.evaluation.planner_edge_gate import (
    PlannerEdgeGateDecision,
    evaluate_planner_edge_gate,
)


def test_planner_edge_gate_passes_when_correlation_meets_threshold() -> None:
    imagined = [5.0, 10.0, -3.0, 1.5, 4.0]
    realised = [4.5, 9.0, -2.5, 1.0, 3.5]

    decision = evaluate_planner_edge_gate(imagined, realised, minimum_correlation=0.2)

    assert isinstance(decision, PlannerEdgeGateDecision)
    assert decision.passed is True
    assert decision.valid_pairs == len(imagined)
    assert decision.correlation is not None
    assert decision.correlation >= 0.2


def test_planner_edge_gate_fails_when_correlation_below_threshold() -> None:
    imagined = [1.0, 2.0, 3.0, 4.0]
    realised = [4.0, 3.0, 2.0, 1.0]

    decision = evaluate_planner_edge_gate(imagined, realised, minimum_correlation=0.5)

    assert decision.passed is False
    assert decision.correlation is not None
    assert decision.correlation < 0.5


def test_planner_edge_gate_skips_invalid_pairs_and_applies_weights() -> None:
    imagined = [1.0, None, 3.0, float("nan"), 5.0]
    realised = [1.2, 2.1, 3.5, 4.0, None]
    weights = [1.0, 2.0, 0.5, 1.0, 1.0]

    decision = evaluate_planner_edge_gate(imagined, realised, weights=weights, minimum_correlation=0.1)

    assert decision.passed is True
    assert decision.valid_pairs == 2
    assert decision.total_pairs == len(imagined)
    assert decision.correlation is not None


def test_planner_edge_gate_requires_sufficient_data() -> None:
    imagined = [1.0, None]
    realised = [2.0, 3.0]

    decision = evaluate_planner_edge_gate(imagined, realised, minimum_correlation=0.2)

    assert decision.passed is False
    assert decision.correlation is None
    assert decision.valid_pairs < 2


def test_planner_edge_gate_validates_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate_planner_edge_gate([1.0, 2.0], [1.5])

    with pytest.raises(ValueError):
        evaluate_planner_edge_gate([], [])

    with pytest.raises(ValueError):
        evaluate_planner_edge_gate([1.0, 2.0], [1.5, 2.5], weights=[1.0])


def test_planner_edge_gate_serialises_to_dict() -> None:
    decision = evaluate_planner_edge_gate([1.0, 2.0, 3.0], [1.1, 2.2, 2.9])

    payload = decision.as_dict()
    assert payload["passed"] is True
    assert payload["minimum_correlation"] == pytest.approx(0.2, rel=1e-9)
    assert payload["valid_pairs"] == 3
    assert payload["total_pairs"] == 3
