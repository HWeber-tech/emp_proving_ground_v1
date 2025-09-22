from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.sensory.anomaly import AnomalySensor
from src.sensory.how.how_sensor import HowSensor


def _build_market_frame(rows: int = 12, *, anomaly_spike: bool = False) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data: list[dict[str, object]] = []
    price = 1.10
    for idx in range(rows):
        price += 0.0005 if idx % 2 == 0 else -0.0003
        if anomaly_spike and idx == rows - 1:
            price += 0.01
        data.append(
            {
                "timestamp": base + timedelta(minutes=idx),
                "symbol": "EURUSD",
                "open": price - 0.0004,
                "high": price + 0.0006,
                "low": price - 0.0005,
                "close": price,
                "volume": 1500 + idx * 120,
                "volatility": 0.0004 + idx * 0.00001,
                "spread": 0.00005,
                "depth": 5500 + idx * 120,
                "order_imbalance": 0.15 + 0.01 * idx,
                "data_quality": 0.9,
            }
        )
    return pd.DataFrame(data)


def test_how_sensor_emits_liquidity_audit() -> None:
    sensor = HowSensor()
    frame = _build_market_frame()

    signals = sensor.process(frame)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "HOW"
    assert -1.0 <= float(signal.value["strength"]) <= 1.0
    metadata = signal.metadata or {}
    assert metadata.get("source") == "sensory.how"
    audit = metadata.get("audit")
    assert isinstance(audit, dict)
    assert set(audit.keys()) >= {"signal", "confidence", "liquidity", "participation"}


def test_anomaly_sensor_sequence_mode_detects_spike() -> None:
    sensor = AnomalySensor()
    frame = _build_market_frame(anomaly_spike=True)

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "ANOMALY"
    metadata = signal.metadata or {}
    assert metadata.get("source") == "sensory.anomaly"
    assert metadata.get("mode") == "sequence"
    assert metadata.get("thresholds") == {"warn": 0.4, "alert": 0.7}
    assert signal.value["strength"] >= 0.0


def test_anomaly_sensor_falls_back_to_market_payload() -> None:
    sensor = AnomalySensor()
    frame = _build_market_frame(rows=4)

    signals = sensor.process(frame)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "ANOMALY"
    metadata = signal.metadata or {}
    assert metadata.get("mode") == "market_data"
    assert "baseline" in metadata.get("audit", {})
