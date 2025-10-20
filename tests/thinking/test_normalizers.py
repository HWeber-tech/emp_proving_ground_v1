"""Regression tests for normalization helpers with defensive guards."""

from __future__ import annotations

import pytest

from src.thinking.models.normalizers import (
    normalize_attack_report,
    normalize_prediction,
    normalize_survival_result,
)


class _ExplodingDict:
    def __init__(self, confidence: float, probability: float) -> None:
        self.confidence = confidence
        self.probability = probability

    def dict(self) -> dict[str, float]:
        raise ValueError("dict() access should be caught")


def test_normalize_prediction_handles_dict_failure() -> None:
    payload = _ExplodingDict(confidence=0.73, probability=0.61)

    normalised = normalize_prediction(payload)

    assert normalised["confidence"] == pytest.approx(0.73)
    assert normalised["probability"] == pytest.approx(0.61)
    assert normalised["actionable"] is False


class _BrokenConfidence:
    @property
    def confidence(self) -> float:  # pragma: no cover - exercised via normalization
        raise RuntimeError("confidence property blew up")

    probability = 0.42


def test_normalize_prediction_handles_attribute_errors() -> None:
    payload = _BrokenConfidence()

    normalised = normalize_prediction(payload)

    assert normalised["confidence"] == 0.0
    assert normalised["probability"] == pytest.approx(0.42)


def test_normalize_prediction_coerces_string_actionable_flag() -> None:
    payload = {
        "confidence": 0.5,
        "probability": 0.4,
        "lower_bound_return": 0.2,
        "upper_bound_return": 0.3,
        "actionable": "false",
    }

    normalised = normalize_prediction(payload)

    assert normalised["actionable"] is False


class _SurvivalProbe:
    @property
    def survival_rate(self) -> float:  # pragma: no cover - exercised via normalization
        raise RuntimeError("not available")

    survival_probability = "0.85"


def test_normalize_survival_result_falls_back_to_probability() -> None:
    probe = _SurvivalProbe()

    snapshot = normalize_survival_result(probe)

    assert snapshot == {"survival_rate": pytest.approx(0.85)}


class _AttackObject:
    def dict(self) -> list[str]:  # pragma: no cover - exercised via normalization
        return ["invalid"]

    attack_id = "atk-1"
    strategy_id = "str-2"
    success = True
    impact = "1.25"
    timestamp = "2024-09-14T10:00:00Z"
    error = None


def test_normalize_attack_report_handles_non_mapping_dict() -> None:
    report = _AttackObject()

    snapshot = normalize_attack_report(report)

    assert snapshot["attack_id"] == "atk-1"
    assert snapshot["strategy_id"] == "str-2"
    assert snapshot["success"] is True
    assert snapshot["impact"] == pytest.approx(1.25)
    assert snapshot["timestamp"] == "2024-09-14T10:00:00Z"
    assert snapshot.get("error", "") == ""
