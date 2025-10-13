from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import pytest

from src.sensory.enhanced.how_dimension import InstitutionalUnderstandingEngine
from src.sensory.how.how_sensor import HowSensor
from src.sensory.organs.dimensions import (
    AnomalySensoryOrgan,
    HowSensoryOrgan,
    WhatSensoryOrgan,
    WhenSensoryOrgan,
    WhySensoryOrgan,
)
from src.sensory.signals import SensorSignal
from src.sensory.why.narrative_hooks import NarrativeEvent


@dataclass
class _StubSignalFactory:
    dimension: str
    strength: float
    confidence: float

    def build(self) -> SensorSignal:
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        metadata = {
            "quality": {
                "source": f"stub.{self.dimension.lower()}",
                "timestamp": timestamp.isoformat(),
                "confidence": self.confidence,
                "strength": self.strength,
            },
            "lineage": {"dimension": self.dimension, "source": f"sensory.{self.dimension.lower()}"},
        }
        value: dict[str, object] = {"confidence": self.confidence}
        if self.dimension == "WHAT":
            value["pattern_strength"] = self.strength
        else:
            value["strength"] = self.strength
        return SensorSignal(
            signal_type=self.dimension,
            value=value,
            confidence=self.confidence,
            timestamp=timestamp,
            metadata=metadata,
        )


class _RecordingWhatSensor:
    def __init__(self, *, strength: float = 0.35, confidence: float = 0.6) -> None:
        self._factory = _StubSignalFactory("WHAT", strength, confidence)
        self.seen_frame: Any = None

    def process(self, frame):  # type: ignore[override]
        self.seen_frame = frame
        return [self._factory.build()]


class _RecordingWhenSensor:
    def __init__(self, *, strength: float = 0.2, confidence: float = 0.55) -> None:
        self._factory = _StubSignalFactory("WHEN", strength, confidence)
        self.seen_frame: Any = None
        self.seen_macro_events: Sequence[datetime] | None = None
        self.seen_option_positions: Any = None

    def process(self, frame, *, option_positions=None, macro_events=None):  # type: ignore[override]
        self.seen_frame = frame
        self.seen_macro_events = list(macro_events or [])
        self.seen_option_positions = option_positions
        return [self._factory.build()]


class _RecordingWhySensor:
    def __init__(self, *, strength: float = 0.4, confidence: float = 0.65) -> None:
        self._factory = _StubSignalFactory("WHY", strength, confidence)
        self.seen_frame: Any = None
        self.seen_events: Sequence[NarrativeEvent] | None = None
        self.seen_flags: Mapping[str, float] | None = None
        self.seen_as_of: datetime | None = None

    def process(self, frame, *, narrative_events=None, macro_regime_flags=None, as_of=None):  # type: ignore[override]
        self.seen_frame = frame
        self.seen_events = list(narrative_events or [])
        self.seen_flags = dict(macro_regime_flags or {})
        self.seen_as_of = as_of
        return [self._factory.build()]


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
async def test_how_sensory_organ_discriminates_market_bias() -> None:
    engine = InstitutionalUnderstandingEngine(random_source=lambda: 0.0)
    sensor = HowSensor(engine=engine)
    organ = HowSensoryOrgan(sensor=sensor)
    timestamp = datetime(2024, 1, 5, 9, tzinfo=timezone.utc)

    bullish_payload = {
        "timestamp": timestamp,
        "symbol": "EURUSD",
        "open": 1.101,
        "high": 1.106,
        "low": 1.099,
        "close": 1.105,
        "volume": 4_800.0,
        "volatility": 0.0006,
        "bid": 1.1048,
        "ask": 1.1051,
        "spread": 0.0003,
        "depth": 18_500.0,
        "order_imbalance": 0.4,
        "data_quality": 0.96,
    }

    bearish_payload = {
        "timestamp": timestamp,
        "symbol": "EURUSD",
        "open": 1.101,
        "high": 1.100,
        "low": 1.090,
        "close": 1.092,
        "volume": 80.0,
        "volatility": 0.12,
        "bid": 1.0900,
        "ask": 1.0975,
        "spread": 0.012,
        "depth": 120.0,
        "order_imbalance": -3.2,
        "data_quality": 0.55,
    }

    bullish_reading = await organ.process(bullish_payload)
    bearish_reading = await organ.process(bearish_payload)

    bullish_strength = bullish_reading.data["signal_strength"]
    bearish_strength = bearish_reading.data["signal_strength"]

    assert bullish_strength > 0.2
    assert bearish_strength < -0.2
    assert bullish_strength > bearish_strength

    bullish_quality = bullish_reading.metadata.get("quality")
    bearish_quality = bearish_reading.metadata.get("quality")
    assert isinstance(bullish_quality, Mapping)
    assert isinstance(bearish_quality, Mapping)
    assert bullish_quality.get("source") == "sensory.how"
    assert bearish_quality.get("source") == "sensory.how"


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


@pytest.mark.asyncio
async def test_what_sensory_organ_wraps_sensor_payload() -> None:
    sensor = _RecordingWhatSensor()
    organ = WhatSensoryOrgan(sensor=sensor)
    payload = {
        "timestamp": datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
        "symbol": "EURUSD",
        "open": 1.101,
        "high": 1.105,
        "low": 1.099,
        "close": 1.104,
        "volume": 2100.0,
    }

    reading = await organ.process(payload)

    assert sensor.seen_frame is not None
    assert list(sensor.seen_frame.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert reading.data["dimension"] == "WHAT"
    assert reading.data["signal_strength"] == pytest.approx(0.35, rel=1e-6)
    value = reading.data["value"]
    assert isinstance(value, Mapping)
    assert value.get("pattern_strength") == pytest.approx(reading.data["signal_strength"], rel=1e-6)
    assert reading.metadata.get("lineage", {}).get("dimension") == "WHAT"
    assert reading.metadata.get("quality", {}).get("source") == "stub.what"


@pytest.mark.asyncio
async def test_when_sensory_organ_merges_macro_context() -> None:
    config = {"macro_events": ["2024-01-02T10:00:00Z"]}
    sensor = _RecordingWhenSensor()
    organ = WhenSensoryOrgan(config=config, sensor=sensor)
    now = datetime(2024, 1, 2, 9, 45, tzinfo=timezone.utc)
    payload = {
        "timestamp": now,
        "symbol": "EURUSD",
        "open": 1.1,
        "high": 1.11,
        "low": 1.098,
        "close": 1.105,
        "volume": 2500.0,
        "macro_events": [now + timedelta(minutes=30)],
        "option_positions": [
            {"strike": 1.1, "gamma": 1200.0},
            {"strike": 1.12, "gamma": -800.0},
        ],
    }

    reading = await organ.process(payload)

    assert sensor.seen_frame is not None
    assert "macro_events" not in sensor.seen_frame.columns
    assert sensor.seen_macro_events is not None
    assert len(sensor.seen_macro_events) == 2
    assert all(event.tzinfo is not None for event in sensor.seen_macro_events)
    assert sensor.seen_macro_events[0] <= sensor.seen_macro_events[1]
    assert sensor.seen_option_positions is not None
    assert getattr(sensor.seen_option_positions, "empty", False) is False
    assert reading.data["dimension"] == "WHEN"
    assert reading.metadata.get("quality", {}).get("source") == "stub.when"


@pytest.mark.asyncio
async def test_why_sensory_organ_injects_narrative_context() -> None:
    baseline_events = [
        {
            "timestamp": "2024-01-02T08:00:00Z",
            "sentiment": 0.3,
            "importance": 0.9,
            "description": "PMI",
        }
    ]
    config = {"narrative_events": baseline_events, "macro_regime_flags": {"growth": 0.2}}
    sensor = _RecordingWhySensor()
    organ = WhySensoryOrgan(config=config, sensor=sensor)

    now = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    payload = {
        "timestamp": now,
        "symbol": "EURUSD",
        "open": 1.09,
        "high": 1.11,
        "low": 1.08,
        "close": 1.105,
        "volume": 1800.0,
        "narrative_events": [
            NarrativeEvent(timestamp=now + timedelta(hours=2), sentiment=-0.4, importance=0.7)
        ],
        "macro_regime_flags": {"inflation": -0.3},
        "as_of": "2024-01-02T09:15:00Z",
    }

    reading = await organ.process(payload)

    assert sensor.seen_frame is not None
    assert "narrative_events" not in sensor.seen_frame.columns
    assert sensor.seen_events is not None
    assert len(sensor.seen_events) == 2
    assert all(isinstance(event, NarrativeEvent) for event in sensor.seen_events)
    assert sensor.seen_flags == {"growth": 0.2, "inflation": -0.3}
    assert sensor.seen_as_of is not None and sensor.seen_as_of.tzinfo is not None
    assert reading.data["dimension"] == "WHY"
    assert reading.metadata.get("quality", {}).get("source") == "stub.why"
