import logging
from datetime import UTC, datetime
from typing import Mapping

import pytest

from src.data_foundation.ingest.telemetry import (
    EventBusIngestPublisher,
    combine_ingest_publishers,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _sample_result() -> TimescaleIngestResult:
    now = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    return TimescaleIngestResult(
        rows_written=10,
        symbols=("EURUSD",),
        start_ts=now,
        end_ts=now,
        ingest_duration_seconds=0.42,
        freshness_seconds=1.5,
        dimension="daily_bars",
        source="yahoo",
    )


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    def publish_from_sync(self, event) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return True


def test_event_bus_publisher_uses_local_bus() -> None:
    bus = _RecordingBus()
    publisher = EventBusIngestPublisher(bus, topic="telemetry.test", source="unit")

    publisher.publish(
        _sample_result(),
        metadata={"window": {"start": datetime(2024, 1, 1, tzinfo=UTC)}},
    )

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.test"
    assert event.source == "unit"
    assert event.payload["result"]["rows_written"] == 10
    assert event.payload["metadata"]["window"]["start"].endswith("+00:00")


def test_event_bus_publisher_falls_back_to_global_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FallbackBus:
        def publish_sync(self, topic: str, payload, source: str | None = None) -> None:
            captured["topic"] = topic
            captured["payload"] = payload
            captured["source"] = source

    class _IdleBus:
        def publish_from_sync(self, event):  # pragma: no cover - branch skipped
            raise RuntimeError("not running")

        def is_running(self) -> bool:
            return False

    monkeypatch.setattr(
        "src.data_foundation.ingest.telemetry.get_global_bus",
        lambda: _FallbackBus(),
    )

    publisher = EventBusIngestPublisher(_IdleBus(), topic="telemetry.fallback", source="unit")
    publisher.publish(_sample_result(), metadata={"latency": 1.23})

    assert captured["topic"] == "telemetry.fallback"
    assert captured["source"] == "unit"
    assert captured["payload"]["result"]["rows_written"] == 10
    assert captured["payload"]["metadata"]["latency"] == 1.23


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[TimescaleIngestResult, Mapping[str, object] | None]] = []

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.calls.append((result, metadata))


def test_combine_ingest_publishers_invokes_all() -> None:
    recorder_one = _Recorder()
    recorder_two = _Recorder()

    combined = combine_ingest_publishers(recorder_one, None, recorder_two)
    assert combined is not None

    metadata = {"reason": "test"}
    combined.publish(_sample_result(), metadata=metadata)

    assert recorder_one.calls and recorder_two.calls
    assert recorder_one.calls[0][1] is metadata
    assert recorder_two.calls[0][1] is metadata


def test_combine_ingest_publishers_handles_none_only() -> None:
    assert combine_ingest_publishers(None) is None

    recorder = _Recorder()
    assert combine_ingest_publishers(None, recorder) is recorder


def test_event_bus_publisher_logs_and_falls_back_on_recoverable_local_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    captured: dict[str, object] = {}

    class _RecoveringBus:
        def publish_from_sync(self, event) -> None:  # pragma: no cover - tested via exception path
            raise RuntimeError("not running")

        def is_running(self) -> bool:
            return True

    class _FallbackBus:
        def publish_sync(self, topic: str, payload, source: str | None = None) -> None:
            captured["topic"] = topic
            captured["payload"] = payload
            captured["source"] = source

    monkeypatch.setattr(
        "src.data_foundation.ingest.telemetry.get_global_bus",
        lambda: _FallbackBus(),
    )

    publisher = EventBusIngestPublisher(_RecoveringBus(), topic="telemetry.recoverable", source="unit")
    publisher.publish(_sample_result(), metadata={"latency": 1.23})

    assert captured["topic"] == "telemetry.recoverable"
    assert "falling back to global bus" in caplog.text


def test_event_bus_publisher_raises_on_unexpected_error() -> None:
    class _BrokenBus:
        def publish_from_sync(self, event) -> None:  # pragma: no cover - tested via exception path
            raise TypeError("boom")

        def is_running(self) -> bool:
            return True

    publisher = EventBusIngestPublisher(_BrokenBus(), topic="telemetry.error", source="unit")

    with pytest.raises(TypeError):
        publisher.publish(_sample_result())
