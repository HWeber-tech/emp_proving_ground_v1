from __future__ import annotations

import logging
import pytest
from datetime import UTC, datetime, timedelta

from src.core.event_bus import Event
from src.data_foundation.persist.timescale import TimescaleIngestRunRecord
from src.operations.event_bus_failover import EventPublishError
from src.operations.ingest_trends import (
    IngestTrendStatus,
    evaluate_ingest_trends,
    publish_ingest_trends,
)


def _record(
    *,
    run_id: str,
    dimension: str = "daily_bars",
    status: str = "ok",
    rows: int = 100,
    freshness: float | None = 60.0,
    executed_at: datetime,
) -> TimescaleIngestRunRecord:
    return TimescaleIngestRunRecord(
        run_id=run_id,
        dimension=dimension,
        status=status,
        rows_written=rows,
        freshness_seconds=freshness,
        ingest_duration_seconds=1.0,
        executed_at=executed_at,
        source="yahoo",
        symbols=("EURUSD",),
        metadata={},
    )


def test_evaluate_ingest_trends_handles_empty_history() -> None:
    snapshot = evaluate_ingest_trends(())

    assert snapshot.status is IngestTrendStatus.warn
    assert "No ingest history" in " ".join(snapshot.issues)
    assert snapshot.dimensions == ()


def test_evaluate_ingest_trends_detects_row_drop() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    records = [
        _record(run_id="r3", rows=40, freshness=90.0, executed_at=base + timedelta(hours=3)),
        _record(run_id="r2", rows=100, freshness=60.0, executed_at=base + timedelta(hours=2)),
        _record(run_id="r1", rows=110, freshness=55.0, executed_at=base + timedelta(hours=1)),
    ]

    snapshot = evaluate_ingest_trends(records)

    assert snapshot.status is IngestTrendStatus.warn
    assert snapshot.dimensions[0].dimension == "daily_bars"
    assert any("Rows dropped" in issue for issue in snapshot.dimensions[0].issues)


def test_publish_ingest_trends_uses_event_bus() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = evaluate_ingest_trends(
        [
            _record(run_id="r1", executed_at=base + timedelta(hours=1)),
            _record(run_id="r0", executed_at=base),
        ]
    )

    class _StubBus:
        def __init__(self) -> None:
            self.events: list[Event] = []

        def is_running(self) -> bool:  # pragma: no cover - trivial
            return True

        def publish_from_sync(self, event: Event) -> int:  # pragma: no cover - trivial
            self.events.append(event)
            return 1

    bus = _StubBus()
    publish_ingest_trends(bus, snapshot)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.ingest.trends"
    assert event.payload["status"] == snapshot.status.value


def test_publish_ingest_trends_logs_failures(monkeypatch, caplog) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = evaluate_ingest_trends(
        [
            _record(run_id="r1", executed_at=base + timedelta(hours=1)),
            _record(run_id="r0", executed_at=base),
        ]
    )

    class _FailRuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> None:
            raise RuntimeError("runtime bus unavailable")

    class _FailGlobalBus:
        def publish_sync(self, *_: object, **__: object) -> None:
            raise RuntimeError("global bus unavailable")

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: _FailGlobalBus(),
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(EventPublishError) as excinfo:
            publish_ingest_trends(_FailRuntimeBus(), snapshot)

    assert "Runtime event bus publish failed; falling back to global bus" in caplog.text
    assert excinfo.value.stage == "global"
    assert excinfo.value.event_type == "telemetry.ingest.trends"


def test_publish_ingest_trends_raises_on_unexpected_runtime_error(caplog) -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = evaluate_ingest_trends(
        [
            _record(run_id="r1", executed_at=base + timedelta(hours=1)),
            _record(run_id="r0", executed_at=base),
        ]
    )

    class _ExplodeRuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> None:
            raise ValueError("boom")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EventPublishError) as excinfo:
            publish_ingest_trends(_ExplodeRuntimeBus(), snapshot)

    assert (
        "Unexpected error publishing ingest trend snapshot via runtime event bus"
        in caplog.text
    )
    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.ingest.trends"
