from __future__ import annotations

import pytest

from src.thinking.learning.horizon_evaluation import (
    HorizonObservation,
    HoldoutCalibrationResult,
    calibrate_holdout_day,
    evaluate_predictions_by_horizon,
)


def _sample_observations() -> list[HorizonObservation]:
    return [
        HorizonObservation(
            "ev1",
            "event-time",
            probability=0.9,
            outcome=1,
            gross_alpha_bps=15,
            fees_bps=5,
        ),
        HorizonObservation(
            1,
            "event-time",
            probability=0.2,
            outcome=0,
            gross_alpha_bps=-5,
            fees_bps=2,
        ),
        HorizonObservation(
            5,
            "event-time",
            probability=0.7,
            outcome=1,
            gross_alpha_bps=10,
            fees_bps=1,
        ),
        HorizonObservation(
            20,
            "event-time",
            probability=0.4,
            outcome=0,
            gross_alpha_bps=-3,
            fees_bps=0.5,
        ),
        HorizonObservation(
            "100ms",
            "wall-time",
            probability=0.3,
            outcome=0,
            gross_alpha_bps=-2,
            fees_bps=0.5,
        ),
        HorizonObservation(
            "0.1s",
            "wall-time",
            probability=0.6,
            outcome=1,
            gross_alpha_bps=12,
            fees_bps=1,
        ),
        HorizonObservation(
            "500ms",
            "wall-time",
            probability=0.4,
            outcome=0,
            gross_alpha_bps=-1,
            fees_bps=0.2,
        ),
        HorizonObservation(
            "2s",
            "wall-time",
            probability=0.8,
            outcome=1,
            gross_alpha_bps=6,
            fees_bps=1.5,
        ),
    ]


def test_evaluate_predictions_by_horizon_reports_metrics() -> None:
    report = evaluate_predictions_by_horizon(_sample_observations(), num_bins=10)

    assert len(report.event_horizons) == 3
    ev1 = report.event_horizons[0]
    assert ev1.horizon == "ev1"
    assert ev1.count == 2
    assert ev1.ece == pytest.approx(0.15, rel=1e-6)
    assert ev1.brier == pytest.approx(0.025, rel=1e-6)
    assert ev1.alpha_after_fees_bps == pytest.approx(1.5, rel=1e-6)
    assert ev1.gross_alpha_bps == pytest.approx(5.0, rel=1e-6)
    assert ev1.fees_bps == pytest.approx(3.5, rel=1e-6)

    ev5 = report.event_horizons[1]
    assert ev5.horizon == "ev5"
    assert ev5.count == 1
    assert ev5.ece == pytest.approx(0.3, rel=1e-6)
    assert ev5.brier == pytest.approx(0.09, rel=1e-6)
    assert ev5.alpha_after_fees_bps == pytest.approx(9.0, rel=1e-6)

    ev20 = report.event_horizons[2]
    assert ev20.horizon == "ev20"
    assert ev20.count == 1
    assert ev20.ece == pytest.approx(0.4, rel=1e-6)
    assert ev20.brier == pytest.approx(0.16, rel=1e-6)
    assert ev20.alpha_after_fees_bps == pytest.approx(-3.5, rel=1e-6)

    assert len(report.time_horizons) == 3
    hundred_ms = report.time_horizons[0]
    assert hundred_ms.horizon == "100ms"
    assert hundred_ms.count == 2
    assert hundred_ms.ece == pytest.approx(0.35, rel=1e-6)
    assert hundred_ms.brier == pytest.approx(0.125, rel=1e-6)
    assert hundred_ms.alpha_after_fees_bps == pytest.approx(4.25, rel=1e-6)

    five_hundred_ms = report.time_horizons[1]
    assert five_hundred_ms.horizon == "500ms"
    assert five_hundred_ms.count == 1
    assert five_hundred_ms.ece == pytest.approx(0.4, rel=1e-6)
    assert five_hundred_ms.brier == pytest.approx(0.16, rel=1e-6)
    assert five_hundred_ms.alpha_after_fees_bps == pytest.approx(-1.2, rel=1e-6)

    two_seconds = report.time_horizons[2]
    assert two_seconds.horizon == "2s"
    assert two_seconds.count == 1
    assert two_seconds.ece == pytest.approx(0.2, rel=1e-6)
    assert two_seconds.brier == pytest.approx(0.04, rel=1e-6)
    assert two_seconds.alpha_after_fees_bps == pytest.approx(4.5, rel=1e-6)

    overall = report.overall
    assert overall.horizon == "all"
    assert overall.horizon_type == "all"
    assert overall.count == 8
    assert overall.ece == pytest.approx(0.2875, rel=1e-6)
    assert overall.brier == pytest.approx(0.09375, rel=1e-6)
    assert overall.alpha_after_fees_bps == pytest.approx(2.5375, rel=1e-6)

    payload = report.as_dict()
    assert payload["overall"]["brier"] == pytest.approx(overall.brier, rel=1e-6)
    assert payload["event_horizons"][0]["horizon"] == "ev1"


def test_evaluate_predictions_by_horizon_validates_inputs() -> None:
    with pytest.raises(ValueError):
        HorizonObservation(
            "ev1",
            "event",
            probability=1.3,
            outcome=0,
        )

    with pytest.raises(ValueError):
        HorizonObservation(
            "ev1",
            "unknown",
            probability=0.5,
            outcome=0,
        )

    with pytest.raises(ValueError):
        HorizonObservation(
            "ev1",
            "event",
            probability=0.5,
            outcome=0,
            weight=0.0,
        )

    observations = _sample_observations()

    with pytest.raises(ValueError):
        evaluate_predictions_by_horizon([], num_bins=10)

    with pytest.raises(ValueError):
        evaluate_predictions_by_horizon(observations, num_bins=0)


def test_calibrate_holdout_day_temperature_scaling_reports_metrics() -> None:
    observations = _sample_observations()

    result = calibrate_holdout_day(
        observations,
        method="temperature",
        num_bins=10,
        label="2024-04-19",
        temperature_bounds=(0.25, 4.0),
        temperature_candidates=101,
    )

    assert isinstance(result, HoldoutCalibrationResult)
    assert result.label == "2024-04-19"
    assert result.method == "temperature"
    assert result.num_observations == len(observations)
    assert result.baseline_brier == pytest.approx(0.09375, rel=1e-6)
    assert result.calibrated_brier == pytest.approx(0.0104730454, rel=1e-6)
    assert result.baseline_ece == pytest.approx(0.2875, rel=1e-6)
    assert result.calibrated_ece == pytest.approx(0.0710062247, rel=1e-6)
    assert result.temperature is not None
    assert 0.25 <= result.temperature <= 4.0


def test_calibrate_holdout_day_isotonic_reports_metrics() -> None:
    observations = _sample_observations()

    result = calibrate_holdout_day(
        observations,
        method="isotonic",
        num_bins=10,
        label="2024-04-19",
    )

    assert isinstance(result, HoldoutCalibrationResult)
    assert result.method == "isotonic"
    assert result.temperature is None
    assert result.calibrated_brier <= result.baseline_brier
    assert result.calibrated_ece <= result.baseline_ece
