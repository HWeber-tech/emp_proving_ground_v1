from __future__ import annotations

import pytest

from src.sensory.thresholds import ThresholdAssessment, evaluate_thresholds


def test_evaluate_thresholds_absolute_alert() -> None:
    assessment = evaluate_thresholds(0.8, warn_threshold=0.3, alert_threshold=0.6)

    assert assessment.state == "alert"
    assert assessment.breached_level == "alert"
    assert assessment.magnitude == pytest.approx(0.8)
    assert assessment.distance_to_alert == 0.0
    assert assessment.distance_to_warn == pytest.approx(0.0)
    assert assessment.breach_ratio == pytest.approx(0.8 / 0.6)


def test_evaluate_thresholds_positive_warning() -> None:
    assessment = evaluate_thresholds(
        -0.7,
        warn_threshold=0.5,
        alert_threshold=0.9,
        mode="positive",
    )

    assert assessment.state == "nominal"
    assert assessment.breached_level is None
    assert assessment.magnitude == 0.0

    assessment = evaluate_thresholds(
        0.55,
        warn_threshold=0.5,
        alert_threshold=0.9,
        mode="positive",
    )

    assert assessment.state == "warning"
    assert assessment.breached_level == "warn"
    assert assessment.distance_to_alert == pytest.approx(0.35)


def test_threshold_assessment_as_dict_contains_expected_fields() -> None:
    assessment = ThresholdAssessment(
        state="warning",
        magnitude=0.4,
        thresholds={"warn": 0.3, "alert": 0.6},
        breached_level="warn",
        breach_ratio=0.4 / 0.6,
        distance_to_warn=0.0,
        distance_to_alert=0.2,
    )

    payload = assessment.as_dict()
    assert payload["state"] == "warning"
    assert payload["thresholds"] == {"warn": 0.3, "alert": 0.6}
    assert payload["breached_level"] == "warn"
