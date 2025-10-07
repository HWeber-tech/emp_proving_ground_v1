from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.sensory.lineage import SensorLineageRecorder, build_lineage_record
from src.sensory.lineage_publisher import SensoryLineagePublisher


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def publish_from_sync(self, event: Any) -> None:
        self.events.append(event)


class _FallbackBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str]] = []

    def publish_sync(self, event_type: str, payload: Any, *, source: str) -> None:
        self.events.append((event_type, payload, source))


def test_build_lineage_record_sanitises_inputs() -> None:
    record = build_lineage_record(
        "HOW",
        "sensory.how",
        inputs={
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "volume": Decimal("123.45"),
        },
        outputs={"signal": 0.5, "confidence": 0.75},
        telemetry={"liquidity": Decimal("0.9")},
        metadata={"mode": "market_data"},
    )

    payload = record.as_dict()
    assert payload["dimension"] == "HOW"
    assert payload["source"] == "sensory.how"
    assert payload["inputs"]["timestamp"].endswith("+00:00")
    assert payload["inputs"]["volume"] == 123.45
    assert payload["outputs"] == {"signal": 0.5, "confidence": 0.75}
    assert payload["telemetry"]["liquidity"] == 0.9
    assert payload["metadata"]["mode"] == "market_data"
    assert "generated_at" in payload


def test_sensor_lineage_recorder_tracks_recent_payloads() -> None:
    recorder = SensorLineageRecorder(max_records=2)

    first = build_lineage_record(
        "HOW",
        "sensory.how",
        inputs={"volume": 1},
        outputs={"signal": 0.1, "confidence": 0.4},
    )
    second = build_lineage_record(
        "ANOMALY",
        "sensory.anomaly",
        outputs={"signal": 0.3, "confidence": 0.6},
    )
    third = build_lineage_record(
        "HOW",
        "sensory.how",
        outputs={"signal": 0.5, "confidence": 0.8},
    )

    recorder.record(first)
    recorder.record(second)
    recorder.record(third)

    history = recorder.history()
    assert [item["dimension"] for item in history] == ["HOW", "ANOMALY"]
    assert recorder.latest()["outputs"]["signal"] == 0.5
    assert recorder.history(limit=1)[0]["dimension"] == "HOW"

    recorder.clear()
    assert recorder.latest() is None
    assert recorder.history() == []


def test_sensory_lineage_publisher_records_history_and_publishes_event() -> None:
    bus = _RecordingBus()
    publisher = SensoryLineagePublisher(event_bus=bus, max_records=2)

    lineage = build_lineage_record(
        "HOW",
        "sensory.how",
        inputs={"symbol": "EURUSD"},
        outputs={"signal": 0.42, "confidence": 0.7},
        telemetry={"liquidity": 0.9},
    )
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    publisher.record(
        "HOW",
        lineage,
        symbol="EURUSD",
        generated_at=timestamp,
        strength=0.42,
        confidence=0.7,
        state="nominal",
        threshold_state="nominal",
        metadata={"audit": {"signal": 0.42}},
    )

    history = publisher.history()
    assert len(history) == 1
    entry = history[0]
    assert entry["dimension"] == "HOW"
    assert entry["symbol"] == "EURUSD"
    assert entry["confidence"] == 0.7
    assert entry["lineage"]["dimension"] == "HOW"
    assert entry["lineage"]["outputs"]["signal"] == 0.42

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.sensory.lineage"
    assert event.payload["symbol"] == "EURUSD"
    assert event.payload["lineage"]["outputs"]["signal"] == 0.42


def test_sensory_lineage_publisher_falls_back_to_publish_sync() -> None:
    bus = _FallbackBus()
    publisher = SensoryLineagePublisher(event_bus=bus, max_records=1)

    publisher.record(
        "ANOMALY",
        {"dimension": "ANOMALY", "outputs": {"signal": 0.3, "confidence": 0.55}},
        symbol="EURUSD",
        confidence=0.55,
    )

    history = publisher.history()
    assert len(history) == 1
    assert history[0]["dimension"] == "ANOMALY"

    assert bus.events
    event_type, payload, source = bus.events[0]
    assert event_type == "telemetry.sensory.lineage"
    assert payload["dimension"] == "ANOMALY"
    assert source == "sensory.lineage_publisher"
