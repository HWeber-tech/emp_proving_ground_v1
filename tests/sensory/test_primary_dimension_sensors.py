from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.sensory.lineage import SensorLineageRecord
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor


def _build_trending_frame(start: datetime, values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, close in enumerate(values):
        ts = start + timedelta(minutes=idx)
        rows.append(
            {
                "timestamp": ts,
                "symbol": "EURUSD",
                "open": close - 0.05,
                "high": close + 0.05,
                "low": close - 0.07,
                "close": close,
                "volume": 1_000 + idx * 10,
                "data_quality": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_what_sensor_emits_lineage_and_quality_metadata() -> None:
    sensor = WhatSensor()
    start = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    closes = [1.1000 + idx * 0.002 for idx in range(32)]
    frame = _build_trending_frame(start, closes)

    signal = sensor.process(frame)[0]

    assert signal.signal_type == "WHAT"
    assert isinstance(signal.lineage, SensorLineageRecord)
    metadata = signal.metadata or {}
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.what"
    assert quality.get("confidence") == signal.confidence
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("dimension") == "WHAT"
    assert lineage.get("metadata", {}).get("mode") == "pattern_analysis"


def test_why_sensor_adds_lineage_and_quality_metadata() -> None:
    sensor = WhySensor()
    start = datetime(2024, 1, 2, 8, tzinfo=timezone.utc)
    closes = [1.2 + idx * 0.001 for idx in range(40)]
    frame = _build_trending_frame(start, closes)
    frame["macro_bias"] = 0.35
    frame["yield_2y"] = 0.02 + frame.index * 0.0001
    frame["yield_10y"] = 0.03 + frame.index * 0.00005

    signal = sensor.process(frame)[0]

    assert signal.signal_type == "WHY"
    assert isinstance(signal.lineage, SensorLineageRecord)
    metadata = signal.metadata or {}
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.why"
    assert quality.get("confidence") == signal.confidence
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("dimension") == "WHY"
    assert lineage.get("metadata", {}).get("mode") == "macro_yield_fusion"


def test_when_sensor_lineage_captures_temporal_components() -> None:
    sensor = WhenSensor()
    start = datetime(2024, 1, 3, 13, tzinfo=timezone.utc)
    closes = [1.3 + idx * 0.0005 for idx in range(4)]
    frame = _build_trending_frame(start, closes)

    signal = sensor.process(
        frame,
        macro_events=[start + timedelta(minutes=45)],
    )[0]

    assert signal.signal_type == "WHEN"
    assert isinstance(signal.lineage, SensorLineageRecord)
    metadata = signal.metadata or {}
    quality = metadata.get("quality")
    assert isinstance(quality, dict)
    assert quality.get("source") == "sensory.when"
    assert quality.get("confidence") == signal.confidence
    lineage = metadata.get("lineage")
    assert isinstance(lineage, dict)
    assert lineage.get("dimension") == "WHEN"
    assert isinstance(lineage.get("metadata", {}).get("active_sessions"), list)
