from __future__ import annotations

import math

import pytest

from src.reflection.trm.ab_validator import (
    SurrogateABValidationResult,
    validate_surrogate_alignment,
)


def test_validate_surrogate_alignment_accepts_within_threshold() -> None:
    result = validate_surrogate_alignment(
        {"sharpe": 1.05, "turnover": 0.92},
        {"sharpe": 1.0, "turnover": 0.9},
    )

    assert isinstance(result, SurrogateABValidationResult)
    assert not result.should_retrain
    assert result.metric_within_threshold is True
    assert result.turnover_within_threshold is True
    assert math.isclose(result.metric_diff_pct, 5.0, rel_tol=1e-6)
    assert math.isclose(result.turnover_diff_pct, 2.2222222, rel_tol=1e-6)
    assert result.notes == ()


def test_validate_surrogate_alignment_flags_retrain_when_outside_threshold() -> None:
    result = validate_surrogate_alignment(
        {"sharpe": 1.2, "turnover": 1.4},
        {"sharpe": 1.0, "turnover": 1.0},
    )

    assert result.should_retrain is True
    assert result.metric_within_threshold is False
    assert result.turnover_within_threshold is False
    assert any("sharpe" in note for note in result.notes)
    assert any("turnover" in note for note in result.notes)


def test_validate_surrogate_alignment_supports_custom_metric_name() -> None:
    result = validate_surrogate_alignment(
        {"alpha": 12.0, "turnover": 0.9},
        {"alpha": 10.0, "turnover": 0.75},
        metric="alpha",
        metric_tolerance_pct=25.0,
        turnover_tolerance_pct=10.0,
    )

    assert result.metric_name == "alpha"
    assert result.metric_within_threshold is True
    assert result.turnover_within_threshold is False
    assert result.should_retrain is True


def test_validate_surrogate_alignment_rejects_missing_metric() -> None:
    with pytest.raises(KeyError):
        validate_surrogate_alignment({"turnover": 1.0}, {"sharpe": 1.0, "turnover": 1.0})

    with pytest.raises(KeyError):
        validate_surrogate_alignment({"sharpe": 1.0, "turnover": 1.0}, {"turnover": 1.0})


def test_validate_surrogate_alignment_validates_metric_name() -> None:
    with pytest.raises(ValueError):
        validate_surrogate_alignment({"sharpe": 1.0, "turnover": 1.0}, {"sharpe": 1.0, "turnover": 1.0}, metric="")
