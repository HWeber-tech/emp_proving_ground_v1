from __future__ import annotations

import pytest

from src.thinking.learning.horizon_evaluation import (
    HorizonObservation,
    evaluate_predictions_by_horizon,
)


def _sample_observations() -> list[HorizonObservation]:
    return [
        HorizonObservation(
            "ev1",
            "event",
            probability=0.9,
            outcome=1,
            gross_alpha_bps=15,
            fees_bps=5,
        ),
        HorizonObservation(
            "ev1",
            "event",
            probability=0.2,
            outcome=0,
            gross_alpha_bps=-5,
            fees_bps=2,
        ),
        HorizonObservation(
            "ev5",
            "event",
            probability=0.7,
            outcome=1,
            gross_alpha_bps=10,
            fees_bps=1,
        ),
        HorizonObservation(
            "1h",
            "time",
            probability=0.3,
            outcome=0,
            gross_alpha_bps=-2,
            fees_bps=0.5,
        ),
        HorizonObservation(
            "1h",
            "time",
            probability=0.6,
            outcome=1,
            gross_alpha_bps=12,
            fees_bps=1,
        ),
        HorizonObservation(
            "4h",
            "time",
            probability=0.4,
            outcome=0,
            gross_alpha_bps=-1,
            fees_bps=0.2,
        ),
    ]


def test_evaluate_predictions_by_horizon_reports_metrics() -> None:
    report = evaluate_predictions_by_horizon(_sample_observations(), num_bins=10)

    assert len(report.event_horizons) == 2
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

    assert len(report.time_horizons) == 2
    one_hour = report.time_horizons[0]
    assert one_hour.horizon == "1h"
    assert one_hour.count == 2
    assert one_hour.ece == pytest.approx(0.35, rel=1e-6)
    assert one_hour.brier == pytest.approx(0.125, rel=1e-6)
    assert one_hour.alpha_after_fees_bps == pytest.approx(4.25, rel=1e-6)

    four_hour = report.time_horizons[1]
    assert four_hour.horizon == "4h"
    assert four_hour.count == 1
    assert four_hour.ece == pytest.approx(0.4, rel=1e-6)
    assert four_hour.brier == pytest.approx(0.16, rel=1e-6)
    assert four_hour.alpha_after_fees_bps == pytest.approx(-1.2, rel=1e-6)

    overall = report.overall
    assert overall.horizon == "all"
    assert overall.horizon_type == "all"
    assert overall.count == 6
    assert overall.ece == pytest.approx(0.2833333333, rel=1e-6)
    assert overall.brier == pytest.approx(0.0916666667, rel=1e-6)
    assert overall.alpha_after_fees_bps == pytest.approx(3.2166666667, rel=1e-6)

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

