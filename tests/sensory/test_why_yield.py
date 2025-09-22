from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker
from src.sensory.why.why_sensor import WhySensor


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "sensory" / "yield_curve_regimes.json"
)
YIELD_CURVE_FIXTURES = json.loads(_FIXTURE_PATH.read_text())


@pytest.mark.parametrize("regime", sorted(YIELD_CURVE_FIXTURES))
def test_yield_slope_tracker_signal_regimes(regime: str) -> None:
    scenario = YIELD_CURVE_FIXTURES[regime]

    tracker = YieldSlopeTracker()
    tracker.update_many(scenario["tenors"].items())

    direction, confidence = tracker.signal()
    expected = scenario["expectations"]

    slope = tracker.slope("2Y", "10Y")
    assert slope is not None
    assert slope == pytest.approx(expected["slope"], abs=1e-6)

    assert direction == pytest.approx(expected["direction"], abs=1e-6)
    assert confidence == pytest.approx(expected["confidence"], rel=0.15, abs=0.05)

    snapshot = tracker.snapshot()
    assert snapshot.direction == pytest.approx(expected["direction"], abs=1e-6)
    assert snapshot.confidence == pytest.approx(expected["confidence"], rel=0.15, abs=0.05)
    if expected["direction"] < 0:
        assert snapshot.regime == "inverted"
        assert snapshot.inversion_risk >= 0.25
    else:
        assert snapshot.regime in {"flat", "modestly_steep", "steep"}


def test_yield_slope_tracker_snapshot_identifies_inversion() -> None:
    tracker = YieldSlopeTracker()
    tracker.update_many({"2Y": 0.045, "5Y": 0.043, "10Y": 0.035, "30Y": 0.038})

    snapshot = tracker.snapshot()

    assert snapshot.direction < 0
    assert snapshot.inversion_risk > 0.0
    assert snapshot.regime == "inverted"
    assert snapshot.slope_2s10s is not None and snapshot.slope_2s10s < 0


def test_why_sensor_blends_macro_and_yield_information() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
            "open": [99, 100, 101, 102, 103, 104],
            "macro_bias": [0.25, 0.2, 0.3, 0.25, 0.35, 0.4],
            "yield_2y": [0.021, 0.022, 0.023, 0.024, 0.025, 0.026],
            "yield_5y": [0.024, 0.025, 0.026, 0.027, 0.028, 0.029],
            "yield_10y": [0.028, 0.029, 0.030, 0.031, 0.032, 0.033],
            "yield_30y": [0.031, 0.032, 0.033, 0.034, 0.035, 0.036],
        }
    )

    sensor = WhySensor()
    signals = sensor.process(df)

    assert len(signals) == 1
    signal = signals[0]

    assert signal.value["strength"] > 0
    assert signal.confidence >= 0.45

    yield_meta = signal.metadata.get("yield_curve")
    assert isinstance(yield_meta, dict)
    assert yield_meta.get("slope_2s10s") is not None
    assert yield_meta.get("regime") in {"modestly_steep", "steep"}
