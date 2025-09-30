from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.disaster_recovery import (
    DisasterRecoveryStatus,
    RecoveryStep,
    format_disaster_recovery_markdown,
    plan_disaster_recovery,
)
from src.operations.failover_drill import (
    FailoverDrillComponent,
    FailoverDrillSnapshot,
    FailoverDrillStatus,
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


def test_plan_disaster_recovery_merges_snapshots(
    _backup_snapshot: BackupReadinessSnapshot,
    _failover_snapshot: FailoverDrillSnapshot,
) -> None:
    report = plan_disaster_recovery(_backup_snapshot, _failover_snapshot)

    assert report.status is DisasterRecoveryStatus.ready
    assert len(report.steps) == 2
    assert {step.name for step in report.steps} == {
        "Backup validation",
        "Failover automation",
    }


def test_plan_disaster_recovery_escalates_status_on_warning(
    _backup_snapshot: BackupReadinessSnapshot,
    _failover_snapshot: FailoverDrillSnapshot,
) -> None:
    degraded_backup = _backup_snapshot
    degraded_backup = BackupReadinessSnapshot(
        service=degraded_backup.service,
        generated_at=degraded_backup.generated_at,
        status=BackupStatus.warn,
        latest_backup_at=degraded_backup.latest_backup_at,
        next_backup_due_at=degraded_backup.next_backup_due_at,
        retention_days=degraded_backup.retention_days,
        issues=("Retention window below target",),
        metadata=degraded_backup.metadata,
    )

    report = plan_disaster_recovery(degraded_backup, _failover_snapshot)

    assert report.status is DisasterRecoveryStatus.degraded
    backup_step = next(step for step in report.steps if step.name == "Backup validation")
    assert isinstance(backup_step, RecoveryStep)
    assert backup_step.status is DisasterRecoveryStatus.degraded
    assert "Retention window" in "\n".join(backup_step.details)


def test_markdown_formatter_includes_sections(
    _backup_snapshot: BackupReadinessSnapshot,
    _failover_snapshot: FailoverDrillSnapshot,
) -> None:
    report = plan_disaster_recovery(_backup_snapshot, _failover_snapshot)
    markdown = format_disaster_recovery_markdown(report)

    assert "# Disaster recovery drill" in markdown
    assert "## Recovery steps" in markdown
    assert "## Backup readiness snapshot" in markdown
    assert "## Failover drill snapshot" in markdown
