from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.trading.execution.backlog_tracker import EventBacklogTracker


def test_backlog_tracker_records_samples_and_breaches() -> None:
    tracker = EventBacklogTracker(threshold_ms=100.0, window=8)
    base = datetime.now(tz=timezone.utc)
    tracker.record(lag_ms=50.0, timestamp=base)
    tracker.record(lag_ms=150.0, timestamp=base + timedelta(milliseconds=10))
    snapshot = tracker.snapshot()
    assert snapshot["samples"] == 2
    assert snapshot["max_lag_ms"] == 150.0
    assert snapshot["breaches"] == 1
    assert snapshot["healthy"] is False
    assert snapshot["worst_breach_ms"] == 150.0
    assert snapshot["last_breach_at"] is not None


def test_backlog_tracker_handles_empty_samples() -> None:
    tracker = EventBacklogTracker()
    snapshot = tracker.snapshot()
    assert snapshot == {
        "samples": 0,
        "threshold_ms": tracker.threshold_ms,
        "max_lag_ms": None,
        "avg_lag_ms": None,
        "breaches": 0,
        "healthy": True,
        "last_breach_at": None,
    }


def test_backlog_tracker_clamps_negative_lag() -> None:
    tracker = EventBacklogTracker(threshold_ms=10.0)
    tracker.record(lag_ms=-5.0, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    snapshot = tracker.snapshot()
    assert snapshot["max_lag_ms"] == 0.0
    assert snapshot["breaches"] == 0
    assert snapshot["healthy"] is True
