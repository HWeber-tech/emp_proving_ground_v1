from datetime import UTC, datetime

import pytest

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import (
    IngestDimensionMetrics,
    IngestMetricsSnapshot,
)
from src.data_foundation.ingest.observability import (
    IngestObservabilityDimension,
    build_ingest_observability_snapshot,
)
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)


@pytest.fixture()
def base_timestamp() -> datetime:
    return datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def _dimension_by_name(
    snapshot_dimensions: tuple[IngestObservabilityDimension, ...],
    name: str,
) -> IngestObservabilityDimension:
    for dimension in snapshot_dimensions:
        if dimension.dimension == name:
            return dimension
    raise AssertionError(f"missing dimension {name}")


def test_build_ingest_observability_snapshot_merges_sources(
    base_timestamp: datetime,
) -> None:
    metrics = IngestMetricsSnapshot(
        generated_at=base_timestamp,
        dimensions=(
            IngestDimensionMetrics(
                dimension="daily_bars",
                rows=12,
                symbols=("AAPL", "MSFT"),
                ingest_duration_seconds=9.5,
                freshness_seconds=60.0,
                source="timescale",
            ),
            IngestDimensionMetrics(
                dimension="macro_events",
                rows=2,
                symbols=("CPI", "NFP"),
                ingest_duration_seconds=5.0,
                freshness_seconds=None,
                source="fred",
            ),
        ),
    )

    health = IngestHealthReport(
        status=IngestHealthStatus.warn,
        generated_at=base_timestamp,
        checks=(
            IngestHealthCheck(
                dimension="daily_bars",
                status=IngestHealthStatus.warn,
                message="Missing GOOG",
                rows_written=10,
                freshness_seconds=120.0,
                expected_symbols=("AAPL", "MSFT", "GOOG"),
                observed_symbols=("AAPL", "MSFT"),
                missing_symbols=("GOOG",),
                ingest_duration_seconds=8.0,
                metadata={"lag_seconds": 42},
            ),
            IngestHealthCheck(
                dimension="intraday_trades",
                status=IngestHealthStatus.ok,
                message="Healthy",
                rows_written=5,
                freshness_seconds=45.0,
                expected_symbols=tuple(),
                observed_symbols=("AAPL",),
                missing_symbols=tuple(),
                ingest_duration_seconds=3.5,
                metadata={},
            ),
        ),
        metadata={"ingest_run": "primary"},
    )

    failover = IngestFailoverDecision(
        should_failover=True,
        status=IngestHealthStatus.error,
        reason="daily_bars status=error",
        generated_at=base_timestamp,
        triggered_dimensions=("daily_bars",),
        optional_triggers=tuple(),
        planned_dimensions=("daily_bars", "intraday_trades"),
        metadata={"attempt": 1},
    )

    recovery_plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=("GOOG",), lookback_days=3, source="yahoo"),
    )
    recovery = IngestRecoveryRecommendation(
        plan=recovery_plan,
        reasons={"daily_bars": "Backfill missing GOOG"},
        missing_symbols={"daily_bars": ("GOOG",)},
    )

    snapshot = build_ingest_observability_snapshot(
        metrics,
        health,
        failover=failover,
        recovery=recovery,
        metadata={"ci_job": "coverage"},
    )

    assert snapshot.generated_at == base_timestamp
    assert snapshot.status is IngestHealthStatus.warn
    assert snapshot.failover is failover
    # Recovery only attaches when the plan is not empty
    assert snapshot.recovery is recovery
    assert snapshot.metadata == {"ingest_run": "primary", "ci_job": "coverage"}

    by_dimension = snapshot.dimensions
    daily = _dimension_by_name(by_dimension, "daily_bars")
    assert daily.rows == 12  # metrics override the health rows
    assert daily.freshness_seconds == pytest.approx(60.0)
    assert daily.ingest_duration_seconds == pytest.approx(9.5)
    assert daily.observed_symbols == ("AAPL", "MSFT")
    assert daily.missing_symbols == ("GOOG",)
    assert daily.metadata["lag_seconds"] == 42
    assert daily.source == "timescale"

    intraday = _dimension_by_name(by_dimension, "intraday_trades")
    assert intraday.rows == 5
    assert intraday.source is None
    assert intraday.message == "Healthy"

    macro = _dimension_by_name(by_dimension, "macro_events")
    assert macro.status is IngestHealthStatus.ok
    assert macro.message == "Metric recorded without health check"
    assert macro.source == "fred"

    degraded = snapshot.degraded_dimensions()
    assert degraded == ("daily_bars",)

    markdown = snapshot.to_markdown()
    assert "Failover triggered: **YES**" in markdown
    assert "Recovery plan dimensions" in markdown
    assert "Missing symbols: daily_bars: GOOG" in markdown


def test_ingest_observability_snapshot_serialises_dimension_metadata(
    base_timestamp: datetime,
) -> None:
    metrics = IngestMetricsSnapshot(
        generated_at=base_timestamp,
        dimensions=(
            IngestDimensionMetrics(
                dimension="intraday_trades",
                rows=0,
                symbols=tuple(),
                ingest_duration_seconds=2.0,
                freshness_seconds=None,
                source=None,
            ),
        ),
    )

    health = IngestHealthReport(
        status=IngestHealthStatus.ok,
        generated_at=base_timestamp,
        checks=(
            IngestHealthCheck(
                dimension="intraday_trades",
                status=IngestHealthStatus.ok,
                message="Recovered",
                rows_written=0,
                freshness_seconds=None,
                expected_symbols=tuple(),
                observed_symbols=tuple(),
                missing_symbols=tuple(),
                ingest_duration_seconds=None,
                metadata={},
            ),
        ),
    )

    snapshot = build_ingest_observability_snapshot(metrics, health)
    payload = snapshot.as_dict()

    assert payload["status"] == "ok"
    assert payload["total_rows"] == 0
    assert payload["dimensions"][0]["dimension"] == "intraday_trades"
    assert payload["dimensions"][0]["message"] == "Recovered"
    assert payload["degraded_dimensions"] == []
