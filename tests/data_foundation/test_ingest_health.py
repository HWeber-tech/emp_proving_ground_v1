from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.health import (
    IngestHealthStatus,
    evaluate_ingest_health,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result(
    *,
    dimension: str,
    rows: int,
    symbols: tuple[str, ...],
    freshness_seconds: float | None,
) -> TimescaleIngestResult:
    now = datetime.now(tz=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=symbols,
        start_ts=now - timedelta(days=1),
        end_ts=now - timedelta(minutes=5),
        ingest_duration_seconds=1.5,
        freshness_seconds=freshness_seconds,
        dimension=dimension,
        source="test",
    )


def test_ingest_health_flags_missing_planned_slice() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"]),
        intraday=IntradayTradeIngestPlan(symbols=["EURUSD"], lookback_days=1),
    )

    results = {
        "daily_bars": _result(
            dimension="daily_bars",
            rows=200,
            symbols=("EURUSD", "GBPUSD"),
            freshness_seconds=60.0,
        )
    }

    report = evaluate_ingest_health(results, plan=plan)

    assert report.status is IngestHealthStatus.error
    checks = {check.dimension: check for check in report.checks}
    assert checks["intraday_trades"].status is IngestHealthStatus.error
    assert "No rows ingested" in checks["intraday_trades"].message


def test_ingest_health_warns_on_freshness_sla_violation() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"]),
    )

    stale = _result(
        dimension="daily_bars",
        rows=50,
        symbols=("EURUSD",),
        freshness_seconds=48 * 60 * 60.0,  # 48 hours
    )

    report = evaluate_ingest_health({"daily_bars": stale}, plan=plan)

    assert report.status is IngestHealthStatus.warn
    daily_check = next(check for check in report.checks if check.dimension == "daily_bars")
    assert daily_check.status is IngestHealthStatus.warn
    assert "Freshness" in daily_check.message


def test_health_report_serialises_to_dict() -> None:
    plan = TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=["EURUSD"]))
    healthy = _result(
        dimension="daily_bars",
        rows=10,
        symbols=("EURUSD",),
        freshness_seconds=30.0,
    )

    report = evaluate_ingest_health({"daily_bars": healthy}, plan=plan)
    payload = report.as_dict()

    assert payload["status"] == "ok"
    assert isinstance(payload["generated_at"], str)
    assert payload["checks"]
    assert payload["metadata"]["planned_dimensions"] == ["daily_bars"]
