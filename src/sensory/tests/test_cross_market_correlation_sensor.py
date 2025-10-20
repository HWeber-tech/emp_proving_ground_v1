from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.sensory.correlation import (
    CrossMarketCorrelationConfig,
    CrossMarketCorrelationSensor,
)


def _build_timestamp_index(length: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=length, freq="T", tz="UTC")


def test_cross_market_sensor_detects_lead_relationship() -> None:
    rng = np.random.default_rng(42)
    samples = 360
    base = pd.Series(rng.normal(scale=0.5, size=samples)).cumsum()
    follower = base.shift(3) + pd.Series(rng.normal(scale=0.05, size=samples))
    venue_c = base.shift(-2) + pd.Series(rng.normal(scale=0.06, size=samples))

    frame = pd.DataFrame(
        {
            "timestamp": _build_timestamp_index(samples),
            "EURUSD@LDN": base.values,
            "EURUSD@NYC": follower.values,
            "EURUSD@TKO": venue_c.values,
        }
    )

    sensor = CrossMarketCorrelationSensor(
        CrossMarketCorrelationConfig(
            window=240,
            max_lag=5,
            min_samples=120,
            min_correlation=0.6,
            top_relationships=3,
        )
    )

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]

    assert signal.signal_type == "CROSS_MARKET_CORRELATION"
    dominant = signal.value["dominant_relationship"]

    assert dominant["leader"] == "EURUSD@LDN"
    assert dominant["follower"] == "EURUSD@NYC"
    assert dominant["lead_steps"] == 3
    assert dominant["strength"] > 0.8
    assert signal.confidence > 0.5

    observed = signal.value["observed_series"]
    assert "EURUSD@LDN" in observed and "EURUSD@NYC" in observed

    quality = signal.metadata.get("quality", {})
    assert quality.get("relationships") >= 1
    assert "lineage" in signal.metadata


def test_cross_market_sensor_accepts_mapping_payload() -> None:
    rng = np.random.default_rng(7)
    samples = 180
    leader = pd.Series(np.sin(np.linspace(0, 6, samples)))
    follower = leader.shift(2) + pd.Series(rng.normal(scale=0.02, size=samples))
    neutral = pd.Series(rng.normal(scale=0.5, size=samples))

    payload = {
        ("BTCUSD", "CME"): leader,
        ("BTCUSD", "BINANCE"): follower,
        ("ETHUSD", "CME"): neutral,
    }

    sensor = CrossMarketCorrelationSensor(
        CrossMarketCorrelationConfig(
            window=160,
            max_lag=4,
            min_samples=80,
            min_correlation=0.5,
            top_relationships=2,
        )
    )

    signal = sensor.process(payload)[0]

    dominant = signal.value["dominant_relationship"]
    assert dominant["leader"] == "BTCUSD::CME"
    assert dominant["follower"] == "BTCUSD::BINANCE"
    assert dominant["lead_steps"] == 2
    assert dominant["strength"] > 0.5

    assert signal.metadata["quality"]["relationships"] >= 1
    assert isinstance(signal.metadata["lineage"], dict)


def test_cross_market_sensor_returns_default_when_insufficient_series() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": _build_timestamp_index(20),
            "ES": np.linspace(0, 1, 20),
        }
    )

    sensor = CrossMarketCorrelationSensor()
    signal = sensor.process(frame)[0]

    assert signal.value["top_relationships"] == []
    assert signal.metadata.get("reason") in {"insufficient_series", "no_significant_relationships"}
    assert signal.confidence == 0.1
    assert isinstance(signal.metadata["lineage"], dict)
