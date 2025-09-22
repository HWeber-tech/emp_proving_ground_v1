from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import pytest

from src.sensory.when.gamma_exposure import (
    GammaExposureAnalyzer,
    GammaExposureAnalyzerConfig,
    GammaExposureDataset,
)
from src.sensory.when.when_sensor import WhenSensor, WhenSensorConfig


def _build_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                datetime(2025, 3, 20, 12, 30, tzinfo=timezone.utc),
                datetime(2025, 3, 20, 12, 30, tzinfo=timezone.utc),
                datetime(2025, 3, 20, 12, 30, tzinfo=timezone.utc),
                datetime(2025, 3, 20, 12, 30, tzinfo=timezone.utc),
            ],
            "symbol": ["EURUSD"] * 4,
            "underlying_price": [1.10, 1.10, 1.10, 1.10],
            "strike": [1.10, 1.095, 1.105, 1.09],
            "gamma": [0.00045, -0.00038, 0.00032, 0.00022],
            "open_interest": [2100, 1800, 1750, 900],
            "contract_multiplier": [100000] * 4,
        }
    )


def test_gamma_analyzer_detects_pin_risk() -> None:
    positions = _build_positions()
    analyzer = GammaExposureAnalyzer(
        GammaExposureAnalyzerConfig(near_fraction=0.01, pressure_normalizer=5.0e5)
    )

    summary = analyzer.summarise(positions, spot_price=1.10)

    assert summary.has_data
    assert summary.flip_risk is True
    assert summary.pin_risk_score > 0.2
    assert 0.0 < summary.impact_score <= 1.0
    assert summary.dominant_strikes
    primary = summary.primary_strike
    assert primary is not None
    assert pytest.approx(1.10, abs=1e-6) == primary.strike
    assert primary.share_of_total > 0.3
    assert {profile.side for profile in summary.dominant_strikes} == {"long", "short"}


def test_when_sensor_combines_components() -> None:
    market_frame = pd.DataFrame(
        {
            "timestamp": [datetime(2025, 3, 20, 12, 30, tzinfo=timezone.utc)],
            "symbol": ["EURUSD"],
            "close": [1.10],
        }
    )

    dataset = GammaExposureDataset(_build_positions())
    sensor = WhenSensor(
        WhenSensorConfig(),
        gamma_dataset=dataset,
    )

    signal = sensor.process(
        market_frame,
        macro_events=[datetime(2025, 3, 20, 13, 0, tzinfo=timezone.utc)],
    )[0]

    assert signal.signal_type == "WHEN"
    assert signal.value["strength"] > 0.0
    assert signal.metadata["components"]["gamma_impact"] > 0.0
    assert signal.metadata["components"]["session_intensity"] >= 0.7
    assert signal.metadata["components"]["gamma_pin_strike"] == pytest.approx(1.10, abs=1e-6)
    dominant = signal.metadata["gamma_dominant_strikes"]
    assert isinstance(dominant, list) and dominant
    assert dominant[0]["share_of_total"] > 0.3
