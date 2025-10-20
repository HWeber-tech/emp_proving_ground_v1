from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.trading.execution.backlog_tracker import EventBacklogTracker


def test_backlog_tracker_records_samples_and_breaches() -> None:
    tracker = EventBacklogTracker(threshold_ms=100.0, window=8)
    base = datetime.now(tz=timezone.utc)
    observation_ok = tracker.record(lag_ms=50.0, timestamp=base)
    assert observation_ok is not None
    assert observation_ok.breach is False
    assert observation_ok.lag_ms == 50.0

    observation_breach = tracker.record(
        lag_ms=150.0, timestamp=base + timedelta(milliseconds=10)
    )
    assert observation_breach is not None
    assert observation_breach.breach is True
    assert observation_breach.lag_ms == 150.0
    assert observation_breach.threshold_ms == 100.0
    snapshot = tracker.snapshot()
    assert snapshot["samples"] == 2
    assert snapshot["max_lag_ms"] == 150.0
    assert snapshot["avg_lag_ms"] == pytest.approx(100.0)
    assert snapshot["p95_lag_ms"] == pytest.approx(145.0)
    assert snapshot["breaches"] == 1
    assert snapshot["breach_rate"] == pytest.approx(0.5)
    assert snapshot["max_breach_streak"] == 1
    assert snapshot["latest_lag_ms"] == pytest.approx(150.0)
    assert snapshot["healthy"] is False
    assert snapshot["worst_breach_ms"] == 150.0
    assert snapshot["last_breach_at"] is not None
    assert snapshot["breach_active"] is True
    assert snapshot["last_recovery_ms"] is None
    assert snapshot["last_recovery_at"] is None


def test_backlog_tracker_handles_empty_samples() -> None:
    tracker = EventBacklogTracker()
    snapshot = tracker.snapshot()
    assert snapshot == {
        "samples": 0,
        "threshold_ms": tracker.threshold_ms,
        "max_lag_ms": None,
        "avg_lag_ms": None,
        "p95_lag_ms": None,
        "breaches": 0,
        "breach_rate": 0.0,
        "max_breach_streak": 0,
        "latest_lag_ms": None,
        "healthy": True,
        "last_breach_at": None,
        "breach_active": False,
        "last_recovery_at": None,
        "last_recovery_ms": None,
    }


def test_backlog_tracker_clamps_negative_lag() -> None:
    tracker = EventBacklogTracker(threshold_ms=10.0)
    tracker.record(lag_ms=-5.0, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
    snapshot = tracker.snapshot()
    assert snapshot["max_lag_ms"] == 0.0
    assert snapshot["breaches"] == 0
    assert snapshot["breach_rate"] == 0.0
    assert snapshot["max_breach_streak"] == 0
    assert snapshot["latest_lag_ms"] == 0.0
    assert snapshot["p95_lag_ms"] == 0.0
    assert snapshot["healthy"] is True
    assert snapshot["breach_active"] is False


def test_backlog_tracker_rolls_window_efficiency() -> None:
    tracker = EventBacklogTracker(threshold_ms=100.0, window=2)
    base = datetime.now(tz=timezone.utc)
    tracker.record(lag_ms=50.0, timestamp=base)
    tracker.record(lag_ms=150.0, timestamp=base + timedelta(milliseconds=10))
    tracker.record(lag_ms=70.0, timestamp=base + timedelta(milliseconds=20))

    snapshot = tracker.snapshot()
    assert snapshot["samples"] == 2
    assert snapshot["max_lag_ms"] == 150.0
    assert snapshot["avg_lag_ms"] == pytest.approx(110.0)
    assert snapshot["p95_lag_ms"] == pytest.approx(146.0)
    assert snapshot["breaches"] == 1
    assert snapshot["breach_rate"] == pytest.approx(0.5)
    assert snapshot["max_breach_streak"] == 1
    assert snapshot["latest_lag_ms"] == pytest.approx(70.0)
    assert snapshot["worst_breach_ms"] == 150.0


def test_backlog_tracker_updates_worst_breach_after_rollover() -> None:
    tracker = EventBacklogTracker(threshold_ms=100.0, window=2)
    base = datetime.now(tz=timezone.utc)
    tracker.record(lag_ms=150.0, timestamp=base)
    tracker.record(lag_ms=120.0, timestamp=base + timedelta(milliseconds=10))
    tracker.record(lag_ms=130.0, timestamp=base + timedelta(milliseconds=20))

    snapshot = tracker.snapshot()
    assert snapshot["breaches"] == 2
    assert snapshot["worst_breach_ms"] == 130.0
    assert snapshot["max_lag_ms"] == 130.0
    assert snapshot["p95_lag_ms"] == pytest.approx(129.5)
    assert snapshot["breach_rate"] == pytest.approx(1.0)
    assert snapshot["max_breach_streak"] == 2
    assert snapshot["latest_lag_ms"] == pytest.approx(130.0)


def test_backlog_tracker_records_recovery_latency() -> None:
    tracker = EventBacklogTracker(threshold_ms=200.0, window=8)
    base = datetime.now(tz=timezone.utc)
    tracker.record(lag_ms=20.0, timestamp=base)
    tracker.record(lag_ms=320.0, timestamp=base + timedelta(milliseconds=10))
    tracker.record(lag_ms=280.0, timestamp=base + timedelta(milliseconds=60))

    recovery_time = base + timedelta(milliseconds=190)
    observation = tracker.record(lag_ms=40.0, timestamp=recovery_time)
    assert observation is not None
    assert observation.breach is False

    snapshot = tracker.snapshot()
    assert snapshot["breach_active"] is False
    assert snapshot["last_recovery_ms"] == pytest.approx(180.0)
    assert snapshot["last_recovery_at"] is not None
    assert snapshot["breaches"] == 2
