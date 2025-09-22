from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.quality import (
    IngestQualityStatus,
    evaluate_ingest_quality,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result(
    *,
    rows: int,
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD"),
    freshness_seconds: float | None = 60.0,
    dimension: str = "daily_bars",
) -> TimescaleIngestResult:
    now = datetime(2024, 1, 3, tzinfo=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=symbols,
        start_ts=now - timedelta(days=2),
        end_ts=now - timedelta(days=1),
        ingest_duration_seconds=1.2,
        freshness_seconds=freshness_seconds,
        dimension=dimension,
        source="test",
    )


def test_ingest_quality_ok_with_full_coverage() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"], lookback_days=1)
    )
    report = evaluate_ingest_quality({"daily_bars": _result(rows=2)}, plan=plan)

    assert report.status is IngestQualityStatus.ok
    check = report.checks[0]
    assert check.status is IngestQualityStatus.ok
    assert check.coverage_ratio == 1.0
    payload = report.as_dict()
    assert payload["status"] == "ok"
    assert payload["checks"][0]["coverage_ratio"] == 1.0


def test_ingest_quality_warns_on_partial_coverage() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"], lookback_days=2)
    )
    report = evaluate_ingest_quality({"daily_bars": _result(rows=3)}, plan=plan)

    assert report.status is IngestQualityStatus.warn
    check = report.checks[0]
    assert check.status is IngestQualityStatus.warn
    assert check.coverage_ratio is not None and check.coverage_ratio < 1.0
    assert any("Coverage" in message for message in check.messages)


def test_ingest_quality_errors_when_no_rows() -> None:
    plan = TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=1))
    report = evaluate_ingest_quality({"daily_bars": _result(rows=0)}, plan=plan)

    assert report.status is IngestQualityStatus.error
    check = report.checks[0]
    assert check.status is IngestQualityStatus.error
    assert check.score == 0.0
    assert "No rows ingested" in " ".join(check.messages)
