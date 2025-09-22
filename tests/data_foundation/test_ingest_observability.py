from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import IngestHealthStatus, evaluate_ingest_health
from src.data_foundation.ingest.metrics import summarise_ingest_metrics
from src.data_foundation.ingest.observability import build_ingest_observability_snapshot
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult


def _result(
    *,
    dimension: str,
    rows: int,
    freshness_seconds: float | None,
    symbols: tuple[str, ...] = ("EURUSD",),
    source: str = "yahoo",
) -> TimescaleIngestResult:
    now = datetime.now(tz=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=symbols,
        start_ts=now - timedelta(minutes=15),
        end_ts=now,
        ingest_duration_seconds=3.2,
        freshness_seconds=freshness_seconds,
        dimension=dimension,
        source=source,
    )


def test_build_ingest_observability_snapshot_includes_failover_and_recovery() -> None:
    results = {
        "daily_bars": _result(
            dimension="daily_bars",
            rows=12,
            freshness_seconds=120.0,
            symbols=("EURUSD",),
        ),
    }

    metrics = summarise_ingest_metrics(results)
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=("EURUSD", "GBPUSD"), lookback_days=30)
    )
    health = evaluate_ingest_health(results, plan=plan)

    failover = IngestFailoverDecision(
        should_failover=False,
        status=health.status,
        reason=None,
        generated_at=health.generated_at,
        triggered_dimensions=tuple(),
        optional_triggers=tuple(),
        planned_dimensions=("daily_bars",),
        metadata=health.metadata,
    )

    recovery = IngestRecoveryRecommendation(
        plan=TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=("GBPUSD",), lookback_days=45)),
        reasons={"daily_bars": "Missing GBPUSD"},
        missing_symbols={"daily_bars": ("GBPUSD",)},
    )

    snapshot = build_ingest_observability_snapshot(
        metrics,
        health,
        failover=failover,
        recovery=recovery,
        metadata={"recovery": {"attempts": 1}},
    )

    assert snapshot.status is IngestHealthStatus.warn
    assert snapshot.total_rows() == metrics.total_rows()
    assert snapshot.failover is failover

    dimension = snapshot.dimensions[0]
    assert dimension.dimension == "daily_bars"
    assert dimension.status is IngestHealthStatus.warn
    assert dimension.missing_symbols == ("GBPUSD",)

    summary = snapshot.recovery_summary()
    assert summary is not None
    assert summary["reasons"]["daily_bars"] == "Missing GBPUSD"

    markdown = snapshot.to_markdown()
    assert "Overall status" in markdown
    assert "daily_bars" in markdown

    payload = snapshot.as_dict()
    assert payload["status"] == "warn"
    assert payload["failover"]["should_failover"] is False
    assert payload["recovery"]["missing_symbols"]["daily_bars"] == ["GBPUSD"]


def test_build_ingest_observability_snapshot_adds_metric_only_dimension() -> None:
    results = {
        "daily_bars": _result(
            dimension="daily_bars",
            rows=7,
            freshness_seconds=90.0,
            symbols=("EURUSD",),
        ),
        "intraday_trades": _result(
            dimension="intraday_trades",
            rows=0,
            freshness_seconds=None,
            symbols=tuple(),
        ),
    }

    metrics = summarise_ingest_metrics(results)
    health = evaluate_ingest_health(results, plan=None)

    snapshot = build_ingest_observability_snapshot(metrics, health)

    assert snapshot.status is IngestHealthStatus.warn
    names = {dimension.dimension for dimension in snapshot.dimensions}
    assert names == {"daily_bars", "intraday_trades"}

    extra = next(
        dimension for dimension in snapshot.dimensions if dimension.dimension == "intraday_trades"
    )
    assert extra.message.startswith("Ingest healthy") or extra.rows == 0
