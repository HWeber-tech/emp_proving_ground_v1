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
