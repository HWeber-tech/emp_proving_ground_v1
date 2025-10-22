from datetime import UTC, datetime, timedelta

import pytest

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.cross_region_failover import (
    CrossRegionComponent,
    CrossRegionFailoverSnapshot,
    CrossRegionStatus,
)
from src.operations.failover_drill import (
    FailoverDrillComponent,
    FailoverDrillSnapshot,
    FailoverDrillStatus,
)
from src.operations.failover_infrastructure import (
    FailoverInfrastructureStatus,
    FailoverResource,
    format_failover_infrastructure_markdown,
    plan_failover_infrastructure,
)


@pytest.fixture()
def _resources() -> tuple[FailoverResource, ...]:
    return (
        FailoverResource(
            name="timescale",\
            primary_location="us-east-1",\
            secondary_location="us-west-2",\
            tier="gold",\
            rto_seconds=300.0,\
            rpo_seconds=60.0,\
            notes=("Mirror replica maintained via streaming replication",),
        ),
        FailoverResource(
            name="redis",\
            primary_location="us-east-1",\
            secondary_location="us-west-2",\
            tier="silver",\
            rto_seconds=120.0,\
            rpo_seconds=30.0,\
            notes=tuple(),
        ),
    )


@pytest.fixture()
def _backup_snapshot() -> BackupReadinessSnapshot:
    now = datetime.now(tz=UTC)
    return BackupReadinessSnapshot(
        service="timescale",
        generated_at=now,
        status=BackupStatus.ok,
        latest_backup_at=now - timedelta(hours=1),
        next_backup_due_at=now + timedelta(hours=5),
        retention_days=14,
        issues=tuple(),
        metadata={
            "policy": {
                "providers": ("aws_s3", "gcs"),
                "storage_location": "s3://emp/timescale",
            }
        },
    )


@pytest.fixture()
def _failover_snapshot() -> FailoverDrillSnapshot:
    now = datetime.now(tz=UTC)
    check = IngestHealthCheck(
        dimension="daily_bars",
        status=IngestHealthStatus.ok,
        message="healthy",
        rows_written=100,
        freshness_seconds=10.0,
        expected_symbols=("SPY",),
        observed_symbols=("SPY",),
        missing_symbols=tuple(),
        ingest_duration_seconds=0.5,
        metadata={"dimension": "daily_bars"},
    )
    report = IngestHealthReport(
        status=IngestHealthStatus.ok,
        generated_at=now,
        checks=(check,),
        metadata={"source": "test"},
    )
    decision = IngestFailoverDecision(
        should_failover=False,
        status=IngestHealthStatus.ok,
        reason=None,
        generated_at=now,
        triggered_dimensions=tuple(),
        optional_triggers=tuple(),
        planned_dimensions=("daily_bars",),
        metadata={},
    )
    component = FailoverDrillComponent(
        name="health",
        status=FailoverDrillStatus.ok,
        summary="healthy",
        metadata={"checks": [check.as_dict()]},
    )
    return FailoverDrillSnapshot(
        status=FailoverDrillStatus.ok,
        generated_at=now,
        scenario="timescale_failover",
        components=(component,),
        health_report=report,
        failover_decision=decision,
        metadata={"fallback": {"executed": False, "error": None}},
    )


@pytest.fixture()
def _cross_region_snapshot() -> CrossRegionFailoverSnapshot:
    now = datetime.now(tz=UTC)
    component = CrossRegionComponent(
        name="replica:daily_bars",
        status=CrossRegionStatus.ok,
        summary="replica healthy",
        metadata={
            "dimension": "daily_bars",
            "lag_seconds": 5.0,
        },
    )
    return CrossRegionFailoverSnapshot(
        status=CrossRegionStatus.ok,
        generated_at=now,
        primary_region="us-east-1",
        replica_region="us-west-2",
        components=(component,),
        metadata={},
    )


def test_plan_failover_infrastructure_ready(
    _resources: tuple[FailoverResource, ...],
    _backup_snapshot: BackupReadinessSnapshot,
    _failover_snapshot: FailoverDrillSnapshot,
    _cross_region_snapshot: CrossRegionFailoverSnapshot,
) -> None:
    plan = plan_failover_infrastructure(
        _resources,
        backup_snapshot=_backup_snapshot,
        failover_snapshot=_failover_snapshot,
        cross_region_snapshot=_cross_region_snapshot,
    )

    assert plan.status is FailoverInfrastructureStatus.ready
    assert len(plan.components) == 4
    assert plan.metadata["resource_count"] == 2


def test_plan_failover_infrastructure_escalates_without_snapshots(
    _resources: tuple[FailoverResource, ...],
) -> None:
    plan = plan_failover_infrastructure(_resources)

    assert plan.status is FailoverInfrastructureStatus.degraded
    degraded_components = {
        component.name: component.status for component in plan.components
    }
    assert degraded_components["Backups"] is FailoverInfrastructureStatus.degraded
    assert degraded_components["Failover automation"] is FailoverInfrastructureStatus.degraded
    assert degraded_components["Cross-region replication"] is FailoverInfrastructureStatus.degraded


def test_markdown_formatter_adds_sections(
    _resources: tuple[FailoverResource, ...],
    _backup_snapshot: BackupReadinessSnapshot,
    _failover_snapshot: FailoverDrillSnapshot,
    _cross_region_snapshot: CrossRegionFailoverSnapshot,
) -> None:
    plan = plan_failover_infrastructure(
        _resources,
        backup_snapshot=_backup_snapshot,
        failover_snapshot=_failover_snapshot,
        cross_region_snapshot=_cross_region_snapshot,
    )
    markdown = format_failover_infrastructure_markdown(plan)

    assert "# Failover infrastructure readiness" in markdown
    assert "## Protected resources" in markdown
    assert "## Components" in markdown
    assert "## Cross-region replication snapshot" in markdown
