from datetime import datetime, timedelta, timezone

import pytest

from src.trading.execution.performance_monitor import ThroughputMonitor


UTC = timezone.utc


def test_throughput_monitor_empty_snapshot() -> None:
    monitor = ThroughputMonitor()

    snapshot = monitor.snapshot()

    assert snapshot["samples"] == 0
    assert snapshot["avg_processing_ms"] is None
    assert snapshot["throughput_per_min"] is None


def test_throughput_monitor_records_samples() -> None:
    monitor = ThroughputMonitor(window=10)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    interval = timedelta(milliseconds=200)
    lag = timedelta(milliseconds=10)

    for idx in range(3):
        started_at = base + idx * interval
        finished_at = started_at + timedelta(milliseconds=50)
        ingested_at = started_at - lag
        monitor.record(
            started_at=started_at,
            finished_at=finished_at,
            ingested_at=ingested_at,
        )

    snapshot = monitor.snapshot()

    assert snapshot["samples"] == 3
    assert snapshot["avg_processing_ms"] == pytest.approx(50.0)
    assert snapshot["p95_processing_ms"] == pytest.approx(50.0)
    assert snapshot["max_processing_ms"] == pytest.approx(50.0)
    assert snapshot["avg_lag_ms"] == pytest.approx(10.0)
    assert snapshot["max_lag_ms"] == pytest.approx(10.0)
    assert snapshot["throughput_per_min"] == pytest.approx(400.0)


def test_throughput_monitor_requires_positive_window() -> None:
    with pytest.raises(ValueError):
        ThroughputMonitor(window=0)


def test_throughput_monitor_snapshot_caches_until_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ThroughputMonitor(window=4)

    base = datetime(2024, 1, 1, tzinfo=UTC)
    monitor.record(
        started_at=base,
        finished_at=base + timedelta(milliseconds=20),
        ingested_at=base - timedelta(milliseconds=5),
    )

    call_count = {"value": 0}

    original = ThroughputMonitor._percentile

    def _counting_percentile(values, percentile):
        call_count["value"] += 1
        return original(values, percentile)

    monkeypatch.setattr(
        ThroughputMonitor,
        "_percentile",
        staticmethod(_counting_percentile),
    )

    snapshot_first = monitor.snapshot()
    assert snapshot_first["samples"] == 1
    assert call_count["value"] == 1

    snapshot_second = monitor.snapshot()
    assert snapshot_second == snapshot_first
    assert call_count["value"] == 1, "cached snapshot should reuse percentile result"

    monitor.record(
        started_at=base + timedelta(milliseconds=40),
        finished_at=base + timedelta(milliseconds=70),
        ingested_at=base + timedelta(milliseconds=35),
    )

    snapshot_third = monitor.snapshot()
    assert snapshot_third["samples"] == 2
    assert call_count["value"] == 2, "new sample should invalidate cached snapshot"
