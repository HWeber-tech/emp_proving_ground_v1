from datetime import UTC, datetime, timedelta

import pytest

from src.data_foundation.ingest.anomaly_detection import (
    FeedAnomalySeverity,
    FalseTickSeverity,
    detect_false_ticks,
    detect_feed_anomalies,
)
from src.data_foundation.ingest.quality import evaluate_ingest_quality
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result(
    *,
    rows: int,
    freshness_seconds: float | None,
    dimension: str = "daily_bars",
) -> TimescaleIngestResult:
    now = datetime(2024, 1, 3, tzinfo=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=("EURUSD",),
        start_ts=now - timedelta(days=2),
        end_ts=now - timedelta(days=1),
        ingest_duration_seconds=1.0,
        freshness_seconds=freshness_seconds,
        dimension=dimension,
        source="test",
    )


def test_detect_feed_anomalies_flags_stalled_feed() -> None:
    anomalies = detect_feed_anomalies({"daily_bars": _result(rows=0, freshness_seconds=None)})

    assert len(anomalies) == 1
    anomaly = anomalies[0]
    assert anomaly.severity is FeedAnomalySeverity.CRITICAL
    assert anomaly.code == "feed_break"
    assert "likely stalled" in anomaly.message


def test_detect_feed_anomalies_warns_on_stale_feed() -> None:
    anomalies = detect_feed_anomalies(
        {"intraday_trades": _result(rows=10, freshness_seconds=3_600.0, dimension="intraday_trades")}
    )

    assert anomalies
    anomaly = anomalies[0]
    assert anomaly.severity in {FeedAnomalySeverity.WARNING, FeedAnomalySeverity.CRITICAL}
    assert anomaly.code == "stale_feed"
    assert "exceeds" in anomaly.message


def test_detect_false_ticks_identifies_outlier() -> None:
    prices = [100.0] * 10 + [150.0] + [100.5] * 5
    anomalies = detect_false_ticks(prices, window=10, z_threshold=4.0, min_relative_jump=0.1)

    assert len(anomalies) == 1
    anomaly = anomalies[0]
    assert anomaly.index == 10
    assert anomaly.severity in {FalseTickSeverity.WARNING, FalseTickSeverity.CRITICAL}
    assert anomaly.metadata["relative_jump"] >= 0.1


def test_evaluate_ingest_quality_includes_anomaly_metadata() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=1)
    )
    report = evaluate_ingest_quality(
        {"daily_bars": _result(rows=0, freshness_seconds=7200.0)},
        plan=plan,
    )

    assert report.metadata["anomalies"], "expected anomalies metadata on report"
    check = report.checks[0]
    assert any("stalled" in message or "exceeds" in message for message in check.messages)
    assert check.metadata.get("anomalies"), "expected anomalies attached to check metadata"


def test_detect_false_ticks_requires_minimum_window() -> None:
    with pytest.raises(ValueError):
        detect_false_ticks([1.0, 2.0, 3.0], window=3)
