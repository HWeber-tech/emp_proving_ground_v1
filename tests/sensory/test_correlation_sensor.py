from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.sensory.correlation import CrossMarketCorrelationSensor


def _build_cross_market_frame(samples: int = 120) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []

    minutes = np.arange(samples, dtype=float)
    trend = minutes * 0.05
    oscillation = 0.35 * np.sin(np.linspace(0.0, 6.0, samples))
    es_prices = 100.0 + trend + oscillation

    # SPY lags ES by one step with minor venue-specific noise
    spy_base = np.roll(es_prices, 1)
    spy_base[0] = es_prices[0]
    spy_prices = spy_base + 0.01 * np.cos(np.linspace(0.0, 4.0, samples))

    # DAX behaves largely independently to act as control
    dax_component = 80.0 + 0.02 * np.sin(np.linspace(0.0, 10.0, samples))

    timestamps = [start + timedelta(minutes=idx) for idx in range(samples)]

    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "timestamp": ts,
                "symbol": "ES",
                "venue": "CME",
                "close": float(es_prices[idx]),
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "symbol": "SPY",
                "venue": "NYSE",
                "close": float(spy_prices[idx]),
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "symbol": "DAX",
                "venue": "XETRA",
                "close": float(dax_component[idx]),
            }
        )

    return pd.DataFrame(rows)


def test_correlation_sensor_detects_leading_relationship() -> None:
    sensor = CrossMarketCorrelationSensor(window=90, max_lag=3, min_overlap=15, smoothing=0.5)
    frame = _build_cross_market_frame()

    signal = sensor.process(frame)[0]

    assert signal.signal_type == "CORRELATION"
    relationships = signal.value.get("relationships")
    assert isinstance(relationships, list)
    assert relationships, "expected at least one relationship"

    top = relationships[0]
    assert top["leader"] == "ES@CME"
    assert top["follower"] == "SPY@NYSE"
    assert top["lag"] >= 1
    assert top["significant"] is True
    assert top["smoothed_correlation"] == pytest.approx(signal.value["strength"], rel=1e-6)
    assert abs(top["smoothed_correlation"]) > 0.85
    assert signal.metadata["quality"]["source"] == "sensory.correlation"
    assert signal.metadata["lineage"]["dimension"] == "CORRELATION"
    assert signal.value["state"] == "active"

    # Second invocation should incorporate historical smoothing
    signal_again = sensor.process(frame)[0]
    second_top = signal_again.value["relationships"][0]
    history = second_top["history"]
    assert history["previous_correlation"] is not None
    assert history["previous_lag"] is not None


def test_correlation_sensor_returns_default_when_series_insufficient() -> None:
    sensor = CrossMarketCorrelationSensor()
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "symbol": ["ES"],
            "close": [100.0],
        }
    )

    signal = sensor.process(frame)[0]

    assert signal.signal_type == "CORRELATION"
    assert signal.value["strength"] == 0.0
    assert signal.value["state"] == "idle"
    assert signal.metadata.get("reason") == "insufficient_series"
    assert signal.metadata["quality"]["reason"] == "insufficient_series"
    assert signal.metadata["quality"]["confidence"] == pytest.approx(signal.confidence)
    assert signal.value["relationships"] == []
