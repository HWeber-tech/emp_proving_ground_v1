from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.lineage import SensorLineageRecord
from src.sensory.how.how_sensor import HowSensor
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


@pytest.mark.asyncio()
async def test_what_sensor_async_context_executes_pattern_engine() -> None:
    sensor = WhatSensor()
    start = datetime(2024, 1, 4, 9, tzinfo=timezone.utc)
    closes = [1.05 + idx * 0.003 for idx in range(64)]
    frame = _build_trending_frame(start, closes)

    signal = sensor.process(frame)[0]

    details = signal.value.get("pattern_details")
    assert isinstance(details, dict)
    assert details, "expected pattern details when running inside async loop"
    payload = (signal.metadata or {}).get("pattern_payload")
    assert isinstance(payload, dict)
    assert payload, "expected pattern payload metadata to be populated"


def test_default_signals_include_quality_and_lineage() -> None:
    what_sensor = WhatSensor()
    when_sensor = WhenSensor()
    why_sensor = WhySensor()

    what_signal = what_sensor.process(None)[0]
    when_signal = when_sensor.process(None)[0]
    why_signal = why_sensor.process(pd.DataFrame())[0]

    for signal, source in [
        (what_signal, "sensory.what"),
        (when_signal, "sensory.when"),
        (why_signal, "sensory.why"),
    ]:
        metadata = signal.metadata or {}
        quality = metadata.get("quality")
        assert isinstance(quality, dict)
        assert quality.get("source") == source
        assert quality.get("confidence") == signal.confidence
        lineage = metadata.get("lineage")
        assert isinstance(lineage, dict)
        assert lineage.get("dimension") == signal.signal_type
        assert lineage.get("metadata", {}).get("mode") == "default"


def test_how_sensor_discriminates_institutional_bias() -> None:
    engine = InstitutionalUnderstandingEngine(random_source=lambda: 0.0)
    sensor = HowSensor(engine=engine)
    timestamp = datetime(2024, 1, 5, 9, tzinfo=timezone.utc)

    bullish_frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "open": 1.101,
                "high": 1.106,
                "low": 1.099,
                "close": 1.105,
                "volume": 4_500.0,
                "volatility": 0.0006,
                "bid": 1.1049,
                "ask": 1.1052,
                "spread": 0.0003,
                "depth": 18_000.0,
                "order_imbalance": 0.35,
                "data_quality": 0.95,
            }
        ]
    )
    bearish_frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "open": 1.101,
                "high": 1.100,
                "low": 1.090,
                "close": 1.092,
                "volume": 50.0,
                "volatility": 0.1,
                "bid": 1.0900,
                "ask": 1.0970,
                "spread": 0.012,
                "depth": 80.0,
                "order_imbalance": -3.5,
                "data_quality": 0.55,
            }
        ]
    )

    bullish_signal = sensor.process(bullish_frame)[0]
    bearish_signal = sensor.process(bearish_frame)[0]

    assert bullish_signal.value["strength"] > 0.25
    assert bearish_signal.value["strength"] < -0.25
    assert bullish_signal.value["strength"] > bearish_signal.value["strength"]


def test_what_sensor_breakout_strength_tracks_extremes() -> None:
    start = datetime(2024, 1, 6, 12, tzinfo=timezone.utc)
    rising = [1.1000, 1.1020, 1.1040, 1.1060, 1.1080]
    falling = [1.1200, 1.1180, 1.1150, 1.1120, 1.1080]

    sensor_up = WhatSensor()
    sensor_up._run_pattern_orchestrator = lambda df: {}
    up_signal = sensor_up.process(_build_trending_frame(start, rising))[0]

    sensor_down = WhatSensor()
    sensor_down._run_pattern_orchestrator = lambda df: {}
    down_signal = sensor_down.process(_build_trending_frame(start, falling))[0]

    assert up_signal.value["pattern_strength"] >= 0.6
    assert down_signal.value["pattern_strength"] <= -0.6
    assert up_signal.value["pattern_strength"] > down_signal.value["pattern_strength"]


def test_when_sensor_macro_events_increase_urgency() -> None:
    sensor = WhenSensor()
    start = datetime(2024, 1, 7, 9, tzinfo=timezone.utc)
    closes = [1.2000 + idx * 0.0004 for idx in range(12)]
    frame = _build_trending_frame(start, closes)

    baseline_signal = sensor.process(frame, macro_events=[])[0]
    imminent_event = frame["timestamp"].iloc[-1] + timedelta(minutes=20)
    with_event_signal = sensor.process(frame, macro_events=[imminent_event])[0]

    assert with_event_signal.value["strength"] > baseline_signal.value["strength"]


def test_why_sensor_yield_curve_modulates_strength() -> None:
    sensor = WhySensor()
    start = datetime(2024, 1, 8, 10, tzinfo=timezone.utc)
    closes = [1.2500 + idx * 0.0002 for idx in range(40)]
    base_frame = _build_trending_frame(start, closes)
    base_frame["macro_bias"] = 0.0

    steep_frame = base_frame.copy()
    steep_frame["yield_2y"] = 0.02
    steep_frame["yield_10y"] = 0.03

    inverted_frame = base_frame.copy()
    inverted_frame["yield_2y"] = 0.035
    inverted_frame["yield_10y"] = 0.025

    steep_signal = sensor.process(steep_frame)[0]
    inverted_signal = sensor.process(inverted_frame)[0]

    assert steep_signal.value["strength"] > inverted_signal.value["strength"]
