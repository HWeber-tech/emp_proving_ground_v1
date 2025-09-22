from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.configuration import TimescaleCrossRegionSettings
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    IngestSchedulerStatus,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult, TimescaleIngestRunRecord
from src.operations.cross_region_failover import (
    CrossRegionStatus,
    evaluate_cross_region_failover,
)
from src.operations.failover_drill import (
    FailoverDrillComponent,
    FailoverDrillSnapshot,
    FailoverDrillStatus,
)
from types import SimpleNamespace


def _make_settings(**overrides) -> TimescaleCrossRegionSettings:
    base = TimescaleCrossRegionSettings(
        enabled=True,
        primary_region="eu-west",
        replica_region="us-east",
        warn_after_seconds=60.0,
        fail_after_seconds=300.0,
        max_row_difference_ratio=0.1,
        dimensions=("daily_bars",),
        replica_settings=None,
    )
    return TimescaleCrossRegionSettings(
        enabled=base.enabled,
        primary_region=overrides.get("primary_region", base.primary_region),
        replica_region=overrides.get("replica_region", base.replica_region),
        warn_after_seconds=overrides.get("warn_after_seconds", base.warn_after_seconds),
        fail_after_seconds=overrides.get("fail_after_seconds", base.fail_after_seconds),
        max_row_difference_ratio=overrides.get(
            "max_row_difference_ratio", base.max_row_difference_ratio
        ),
        max_schedule_interval_seconds=overrides.get(
            "max_schedule_interval_seconds", base.max_schedule_interval_seconds
        ),
        dimensions=overrides.get("dimensions", base.dimensions),
        replica_settings=overrides.get("replica_settings", base.replica_settings),
    )


def _make_primary(rows: int = 10) -> TimescaleIngestResult:
    now = datetime.now(tz=UTC)
    return TimescaleIngestResult(
        rows_written=rows,
        symbols=("EURUSD",),
        start_ts=now - timedelta(days=1),
        end_ts=now,
        ingest_duration_seconds=1.2,
        freshness_seconds=10.0,
        dimension="daily_bars",
        source="yahoo",
    )


def _make_replica(
    *,
    executed_at: datetime,
    rows: int = 10,
    status: str = "ok",
) -> TimescaleIngestRunRecord:
    return TimescaleIngestRunRecord(
        run_id="replica-1",
        dimension="daily_bars",
        status=status,
        rows_written=rows,
        freshness_seconds=12.0,
        ingest_duration_seconds=1.0,
        executed_at=executed_at,
        source="replica",
        symbols=("EURUSD",),
        metadata={},
    )


def test_evaluate_cross_region_ok_status():
    generated_at = datetime.now(tz=UTC)
    replica = _make_replica(executed_at=generated_at - timedelta(seconds=10))
    snapshot = evaluate_cross_region_failover(
        generated_at=generated_at,
        settings=_make_settings(),
        primary_results={"daily_bars": _make_primary()},
        replica_records={"daily_bars": replica},
        scheduler_snapshot=IngestSchedulerSnapshot(
            status=IngestSchedulerStatus.ok,
            generated_at=generated_at,
            enabled=True,
            running=True,
            consecutive_failures=0,
            interval_seconds=30.0,
            jitter_seconds=5.0,
            max_failures=3,
        ),
        schedule_metadata={"schedule_enabled": True, "schedule_interval_seconds": 30.0},
    )

    assert snapshot.status is CrossRegionStatus.ok
    component = next(comp for comp in snapshot.components if comp.name.startswith("replica"))
    assert component.status is CrossRegionStatus.ok


def test_cross_region_warns_on_lag():
    generated_at = datetime.now(tz=UTC)
    replica = _make_replica(
        executed_at=generated_at - timedelta(seconds=120),
    )
    snapshot = evaluate_cross_region_failover(
        generated_at=generated_at,
        settings=_make_settings(),
        primary_results={"daily_bars": _make_primary()},
        replica_records={"daily_bars": replica},
        scheduler_snapshot=None,
        schedule_metadata={"schedule_enabled": True, "schedule_interval_seconds": 90.0},
    )

    assert snapshot.status is CrossRegionStatus.warn
    lag_component = next(comp for comp in snapshot.components if comp.name.startswith("replica"))
    assert "lag" in lag_component.summary


def test_cross_region_fails_without_replica_history():
    generated_at = datetime.now(tz=UTC)
    failover = FailoverDrillSnapshot(
        status=FailoverDrillStatus.ok,
        generated_at=generated_at,
        scenario="drill",
        components=(
            FailoverDrillComponent(
                name="health",
                status=FailoverDrillStatus.ok,
                summary="healthy",
            ),
        ),
        health_report=SimpleNamespace(as_dict=lambda: {}),
        failover_decision=SimpleNamespace(as_dict=lambda: {}),
    )

    snapshot = evaluate_cross_region_failover(
        generated_at=generated_at,
        settings=_make_settings(),
        primary_results={"daily_bars": _make_primary()},
        replica_records={},
        scheduler_snapshot=IngestSchedulerSnapshot(
            status=IngestSchedulerStatus.warn,
            generated_at=generated_at,
            enabled=False,
            running=False,
            consecutive_failures=1,
            interval_seconds=45.0,
            jitter_seconds=5.0,
            max_failures=3,
            issues=("Scheduler disabled",),
        ),
        schedule_metadata={"schedule_enabled": False, "schedule_interval_seconds": None},
        failover_snapshot=failover,
        replica_error="replica_not_configured",
    )

    assert snapshot.status is CrossRegionStatus.fail
    names = {component.name for component in snapshot.components}
    assert "replica_connection" in names
    scheduler_component = next(comp for comp in snapshot.components if comp.name == "scheduler")
    assert scheduler_component.status in {CrossRegionStatus.warn, CrossRegionStatus.fail}
