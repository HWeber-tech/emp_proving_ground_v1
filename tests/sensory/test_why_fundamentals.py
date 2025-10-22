from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.sensory.why.why_sensor import WhySensor


def _build_trending_frame(start: datetime, closes: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, close in enumerate(closes):
        ts = start + timedelta(minutes=idx)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.985,
                "close": close,
                "volume": 1_000_000 + idx * 5_000,
                "macro_bias": 0.15,
            }
        )
    return pd.DataFrame(rows)


def test_why_sensor_incorporates_fundamental_snapshot() -> None:
    sensor = WhySensor()
    start = datetime(2024, 3, 4, 14, tzinfo=timezone.utc)
    closes = [120.0 + idx * 0.4 for idx in range(40)]
    frame = _build_trending_frame(start, closes)

    fundamentals = {
        "eps": 8.5,
        "book_value_per_share": 65.0,
        "dividend_yield": 0.024,
        "free_cash_flow": 6_500_000.0,
        "shares_outstanding": 520_000.0,
        "growth_rate": 0.09,
        "discount_rate": 0.12,
        "quality_score": 0.72,
    }

    signal = sensor.process(frame, fundamental_snapshot=fundamentals)[0]

    fundamentals_meta = signal.metadata.get("fundamentals")
    assert isinstance(fundamentals_meta, dict)
    metrics = fundamentals_meta.get("metrics")
    assert isinstance(metrics, dict)
    expected_price = frame["close"].iloc[-1]
    assert metrics.get("pe_ratio") == pytest.approx(expected_price / fundamentals["eps"], rel=1e-6)
    assert fundamentals_meta.get("strength", 0.0) > 0.1

    baseline = WhySensor().process(frame)[0]
    assert signal.value["strength"] > baseline.value["strength"]

    assert signal.value["fundamental_strength"] == pytest.approx(fundamentals_meta["strength"], rel=1e-6)
    assert signal.metadata["quality"]["fundamental_confidence"] == pytest.approx(
        fundamentals_meta["confidence"], rel=1e-6
    )
    assert signal.lineage.inputs["fundamental_strength"] == pytest.approx(
        fundamentals_meta["strength"], rel=1e-6
    )


def test_why_sensor_penalises_weak_fundamentals() -> None:
    sensor = WhySensor()
    start = datetime(2024, 3, 5, 10, tzinfo=timezone.utc)
    closes = [95.0 + idx * 0.2 for idx in range(40)]
    frame = _build_trending_frame(start, closes)
    frame["macro_bias"] = 0.05

    fundamentals = {
        "eps": -1.2,
        "free_cash_flow": -2_500_000.0,
        "shares_outstanding": 400_000.0,
        "dividend_yield": 0.0,
        "growth_rate": -0.03,
        "discount_rate": 0.12,
        "quality_score": 0.25,
    }

    signal = sensor.process(frame, fundamental_snapshot=fundamentals)[0]
    fundamentals_meta = signal.metadata.get("fundamentals")
    assert isinstance(fundamentals_meta, dict)
    assert fundamentals_meta.get("strength", 0.0) < -0.1
    assert signal.value["fundamental_strength"] < -0.1

    baseline = WhySensor().process(frame)[0]
    assert signal.value["strength"] < baseline.value["strength"]

    quality = signal.metadata.get("quality", {})
    assert quality.get("fundamental_confidence") == pytest.approx(
        fundamentals_meta.get("confidence", 0.0),
        rel=1e-6,
    )
