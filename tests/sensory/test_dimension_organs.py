from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.sensory.organs.dimensions import AnomalySensoryOrgan, HowSensoryOrgan


@pytest.mark.asyncio
async def test_how_sensory_organ_emits_lineage_and_telemetry() -> None:
    organ = HowSensoryOrgan()
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    payload = {
        "timestamp": now,
        "symbol": "EURUSD",
        "open": 1.1032,
        "high": 1.1051,
        "low": 1.1018,
        "close": 1.1045,
        "volume": 2500.0,
        "volatility": 0.0007,
        "spread": 0.00005,
        "depth": 7200.0,
        "order_imbalance": 0.22,
        "data_quality": 0.92,
        "bid": 1.1042,
        "ask": 1.1047,
    }

    reading = await organ.process(payload)

    assert reading.organ_name == "how_organ"
    assert reading.data["dimension"] == "HOW"
    assert -1.0 <= reading.data["signal_strength"] <= 1.0
    metadata = reading.metadata
    assert metadata.get("threshold_state") in {"nominal", "warning", "alert"}
    telemetry = metadata.get("telemetry")
    assert isinstance(telemetry, dict)
    assert "liquidity" in telemetry
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("dimension") == "HOW"
    assert lineage.get("metadata", {}).get("mode") == "market_data"


@pytest.mark.asyncio
async def test_anomaly_sensory_organ_accumulates_sequence_history() -> None:
    organ = AnomalySensoryOrgan()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    reading = None
    for idx in range(12):
        payload = {
            "timestamp": base_time + timedelta(minutes=idx),
            "symbol": "EURUSD",
            "open": 1.101 + idx * 0.0003,
            "high": 1.102 + idx * 0.0003,
            "low": 1.100 + idx * 0.0003,
            "close": 1.1015 + idx * 0.0004,
            "volume": 1800.0 + idx * 25,
            "volatility": 0.0004 + idx * 0.00002,
            "spread": 0.00004,
        }
        reading = await organ.process(payload)

    assert reading is not None
    assert reading.data["dimension"] == "ANOMALY"
    assert 0.0 <= reading.data["signal_strength"] <= 1.0
    metadata = reading.metadata
    assert metadata.get("threshold_state") in {"nominal", "warning", "alert"}
    telemetry = metadata.get("telemetry")
    assert isinstance(telemetry, dict)
    assert "baseline" in telemetry
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("metadata", {}).get("mode") == "sequence"


@pytest.mark.asyncio
async def test_anomaly_sensory_organ_accepts_sequence_payload() -> None:
    organ = AnomalySensoryOrgan()
    sequence = [1.0, 1.01, 1.015, 1.03, 1.02, 1.025, 1.05, 1.08, 1.1]

    reading = await organ.process(sequence)

    assert reading.data["dimension"] == "ANOMALY"
    assert 0.0 <= reading.data["signal_strength"] <= 1.0
    metadata = reading.metadata
    assert metadata.get("threshold_state") in {"nominal", "warning", "alert"}
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("metadata", {}).get("mode") == "sequence"

