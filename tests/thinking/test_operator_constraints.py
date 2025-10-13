from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.governance.policy_ledger import PolicyLedgerStage
from src.thinking.adaptation.operator_constraints import (
    OperatorContext,
    OperatorConstraint,
    OperatorConstraintSet,
    parse_operator_constraints,
)


def _context(
    *,
    stage: PolicyLedgerStage,
    regime: str,
    confidence: float,
    features: dict[str, float],
    parameters: dict[str, float],
) -> OperatorContext:
    return OperatorContext(
        operation="register_variant",
        stage=stage,
        regime=regime,
        regime_confidence=confidence,
        regime_features=features,
        parameters=parameters,
        metadata={"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()},
    )


def test_operator_constraint_allows_matching_stage_and_regime() -> None:
    constraint = OperatorConstraint(
        name="paper-bull",
        operations=("register_variant",),
        allowed_stages=(PolicyLedgerStage.PAPER,),
        allowed_regimes=("bull",),
        feature_gates={"volatility": {"maximum": 0.5}},
        parameter_bounds={"lookback": (10, 30)},
    )
    context = _context(
        stage=PolicyLedgerStage.PAPER,
        regime="bull",
        confidence=0.7,
        features={"volatility": 0.3},
        parameters={"lookback": 20.0},
    )
    allowed, violations = OperatorConstraintSet((constraint,)).validate(context)
    assert allowed is True
    assert violations == ()


def test_operator_constraint_collects_violation_details() -> None:
    constraint = OperatorConstraint(
        name="paper-bull",
        operations=("register_variant",),
        allowed_stages=(PolicyLedgerStage.PAPER,),
        allowed_regimes=("bull",),
        min_confidence=0.6,
        feature_gates={"volatility": {"maximum": 0.5}},
        parameter_bounds={"lookback": {"minimum": 10, "maximum": 30}},
    )
    context = _context(
        stage=PolicyLedgerStage.PILOT,
        regime="bear",
        confidence=0.4,
        features={"volatility": 0.8},
        parameters={"lookback": 35.0},
    )
    allowed, violations = OperatorConstraintSet((constraint,)).validate(context)
    assert allowed is False
    assert len(violations) == 1
    violation = violations[0]
    assert violation.reason == "operator_constraint_rejected"
    assert violation.details["stage"] == "pilot"
    assert "allowed_stages" in violation.details
    parameters = violation.details["parameters"]
    assert parameters["lookback"]["value"] == pytest.approx(35.0)
    features = violation.details["features"]
    assert features["volatility"]["value"] == pytest.approx(0.8)


def test_parse_operator_constraints_mapping() -> None:
    config = {
        "paper-only": {
            "operations": ["register_variant"],
            "allowed_stages": ["paper"],
        }
    }
    constraint_set = parse_operator_constraints(config)
    assert isinstance(constraint_set, OperatorConstraintSet)
    context = _context(
        stage=PolicyLedgerStage.PAPER,
        regime="neutral",
        confidence=0.5,
        features={},
        parameters={},
    )
    allowed, violations = constraint_set.validate(context)
    assert allowed is True
    assert violations == ()
