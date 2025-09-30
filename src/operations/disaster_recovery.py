"""Disaster recovery helpers built on top of backup and failover telemetry."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Mapping, Sequence

from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult

from .backup import (
    BackupPolicy,
    BackupReadinessSnapshot,
    BackupState,
    BackupStatus,
    evaluate_backup_readiness,
)
from .failover_drill import (
    FailoverDrillSnapshot,
    FailoverDrillStatus,
    execute_failover_drill,
)


class DisasterRecoveryStatus(StrEnum):
    """Aggregated readiness derived from backup and failover posture."""

    ready = "ready"
    degraded = "degraded"
    blocked = "blocked"


_STATUS_ORDER: dict[DisasterRecoveryStatus, int] = {
    DisasterRecoveryStatus.ready: 0,
    DisasterRecoveryStatus.degraded: 1,
    DisasterRecoveryStatus.blocked: 2,
}

_BACKUP_TO_RECOVERY: dict[BackupStatus, DisasterRecoveryStatus] = {
    BackupStatus.ok: DisasterRecoveryStatus.ready,
    BackupStatus.warn: DisasterRecoveryStatus.degraded,
    BackupStatus.fail: DisasterRecoveryStatus.blocked,
}

_FAILOVER_TO_RECOVERY: dict[FailoverDrillStatus, DisasterRecoveryStatus] = {
    FailoverDrillStatus.ok: DisasterRecoveryStatus.ready,
    FailoverDrillStatus.warn: DisasterRecoveryStatus.degraded,
    FailoverDrillStatus.fail: DisasterRecoveryStatus.blocked,
}


def _combine_status(
    current: DisasterRecoveryStatus, candidate: DisasterRecoveryStatus
) -> DisasterRecoveryStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class RecoveryStep:
    """Individual step recorded in a disaster recovery drill."""

    name: str
    status: DisasterRecoveryStatus
    summary: str
    details: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "details": list(self.details),
        }


@dataclass(frozen=True)
class DisasterRecoveryReport:
    """Aggregated report combining backup readiness and failover drills."""

    status: DisasterRecoveryStatus
    generated_at: datetime
    steps: tuple[RecoveryStep, ...]
    backup_snapshot: BackupReadinessSnapshot
    failover_snapshot: FailoverDrillSnapshot
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "steps": [step.as_dict() for step in self.steps],
            "backup_snapshot": self.backup_snapshot.as_dict(),
            "failover_snapshot": self.failover_snapshot.as_dict(),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Disaster recovery drill",
            "",
            f"- Status: {self.status.value.upper()}",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Backup status: {self.backup_snapshot.status.value.upper()}",
            f"- Failover status: {self.failover_snapshot.status.value.upper()}",
            "",
        ]

        if self.steps:
            lines.append("## Recovery steps")
            lines.append("| Step | Status | Summary |")
            lines.append("| --- | --- | --- |")
            for step in self.steps:
                lines.append(
                    f"| {step.name} | {step.status.value.upper()} | {step.summary} |"
                )
            lines.append("")

        if any(step.details for step in self.steps):
            lines.append("## Step details")
            for step in self.steps:
                if not step.details:
                    continue
                lines.append(f"### {step.name}")
                for detail in step.details:
                    lines.append(f"- {detail}")
                lines.append("")

        lines.append("## Backup readiness snapshot")
        lines.append(self.backup_snapshot.to_markdown())
        lines.append("")
        lines.append("## Failover drill snapshot")
        lines.append(self.failover_snapshot.to_markdown())

        return "\n".join(lines)


def format_disaster_recovery_markdown(report: DisasterRecoveryReport) -> str:
    """Convenience wrapper to mirror other operational formatters."""

    return report.to_markdown()


def _build_backup_step(snapshot: BackupReadinessSnapshot) -> RecoveryStep:
    status = _BACKUP_TO_RECOVERY[snapshot.status]
    if snapshot.issues:
        summary = f"Backup posture {snapshot.status.value}; review issues"
        details = tuple(snapshot.issues)
    else:
        summary = "Backups meet policy targets"
        details = tuple()

    metadata = snapshot.metadata.get("policy") if isinstance(snapshot.metadata, Mapping) else None
    if metadata and metadata.get("providers"):
        providers = ", ".join(str(provider) for provider in metadata["providers"])
        details += (f"Providers: {providers}",)
    if metadata and metadata.get("storage_location"):
        details += (f"Storage: {metadata['storage_location']}",)

    latest = (
        snapshot.latest_backup_at.isoformat()
        if snapshot.latest_backup_at is not None
        else "not recorded"
    )
    next_due = (
        snapshot.next_backup_due_at.isoformat()
        if snapshot.next_backup_due_at is not None
        else "not scheduled"
    )
    details += (f"Latest backup: {latest}", f"Next due: {next_due}")

    return RecoveryStep(
        name="Backup validation",
        status=status,
        summary=summary,
        details=details,
    )


def _build_failover_step(snapshot: FailoverDrillSnapshot) -> RecoveryStep:
    status = _FAILOVER_TO_RECOVERY[snapshot.status]
    details: list[str] = []
    for component in snapshot.components:
        details.append(
            f"{component.name}: {component.status.value} – {component.summary}"
        )

    fallback = snapshot.metadata.get("fallback") if isinstance(snapshot.metadata, Mapping) else None
    if isinstance(fallback, Mapping):
        executed = fallback.get("executed")
        if executed:
            details.append("Fallback executed during drill")
        elif executed is False:
            details.append("Fallback not executed")
        error = fallback.get("error")
        if error:
            details.append(f"Fallback error: {error}")

    summary = "Failover drill completed"
    if snapshot.status is FailoverDrillStatus.warn:
        summary = "Failover drill completed with warnings"
    elif snapshot.status is FailoverDrillStatus.fail:
        summary = "Failover automation failed"

    return RecoveryStep(
        name="Failover automation",
        status=status,
        summary=summary,
        details=tuple(details),
    )


def plan_disaster_recovery(
    backup_snapshot: BackupReadinessSnapshot,
    failover_snapshot: FailoverDrillSnapshot,
    *,
    generated_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DisasterRecoveryReport:
    """Fuse backup and failover telemetry into a recovery report."""

    moment = generated_at or datetime.now(tz=UTC)
    steps = [
        _build_backup_step(backup_snapshot),
        _build_failover_step(failover_snapshot),
    ]

    status = DisasterRecoveryStatus.ready
    for step in steps:
        status = _combine_status(status, step.status)

    payload_metadata = dict(metadata) if metadata else {}
    payload_metadata.setdefault("scenario", failover_snapshot.scenario)

    return DisasterRecoveryReport(
        status=status,
        generated_at=moment,
        steps=tuple(steps),
        backup_snapshot=backup_snapshot,
        failover_snapshot=failover_snapshot,
        metadata=payload_metadata,
    )


async def _execute_drill(
    *,
    plan: TimescaleBackbonePlan,
    ingest_results: Mapping[str, TimescaleIngestResult],
    fail_dimensions: Sequence[str],
    backup_policy: BackupPolicy,
    backup_state: BackupState,
    scenario: str,
    now: datetime,
    metadata: Mapping[str, object] | None,
) -> DisasterRecoveryReport:
    async def _fallback() -> None:
        return None

    failover_snapshot = await execute_failover_drill(
        plan=plan,
        results=ingest_results,
        fail_dimensions=tuple(fail_dimensions),
        scenario=scenario,
        fallback=_fallback,
        metadata=metadata,
    )
    backup_snapshot = evaluate_backup_readiness(
        backup_policy,
        backup_state,
        service=f"{scenario}_backups",
        now=now,
        metadata={"policy": backup_policy.__dict__},
    )
    return plan_disaster_recovery(
        backup_snapshot,
        failover_snapshot,
        generated_at=now,
        metadata=metadata,
    )


def simulate_disaster_recovery_drill(
    *,
    plan: TimescaleBackbonePlan,
    ingest_results: Mapping[str, TimescaleIngestResult],
    fail_dimensions: Sequence[str],
    backup_policy: BackupPolicy,
    backup_state: BackupState,
    scenario: str = "timescale_failover",
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DisasterRecoveryReport:
    """Execute a failover drill and evaluate disaster recovery posture."""

    moment = now or datetime.now(tz=UTC)
    return asyncio.run(
        _execute_drill(
            plan=plan,
            ingest_results=ingest_results,
            fail_dimensions=tuple(fail_dimensions),
            backup_policy=backup_policy,
            backup_state=backup_state,
            scenario=scenario,
            now=moment,
            metadata=metadata,
        )
    )


def simulate_default_disaster_recovery(
    *,
    scenario: str = "timescale_failover",
    fail_dimensions: Sequence[str] = ("daily_bars",),
    now: datetime | None = None,
) -> DisasterRecoveryReport:
    """Run a representative drill using fixture ingest telemetry."""

    moment = now or datetime.now(tz=UTC)
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["SPY", "QQQ", "ES"], lookback_days=5),
        intraday=IntradayTradeIngestPlan(symbols=["SPY", "QQQ"], lookback_days=2),
    )

    ingest_results = {
        "daily_bars": TimescaleIngestResult(
            rows_written=1800,
            symbols=("SPY", "QQQ", "ES"),
            start_ts=moment - timedelta(days=5),
            end_ts=moment,
            ingest_duration_seconds=18.4,
            freshness_seconds=2_400.0,
            dimension="daily_bars",
            source="timescale",
        ),
        "intraday_trades": TimescaleIngestResult(
            rows_written=96_000,
            symbols=("SPY", "QQQ"),
            start_ts=moment - timedelta(hours=12),
            end_ts=moment,
            ingest_duration_seconds=25.1,
            freshness_seconds=240.0,
            dimension="intraday_trades",
            source="timescale",
        ),
    }

    backup_policy = BackupPolicy(
        expected_frequency_seconds=6 * 60 * 60,
        retention_days=14,
        minimum_retention_days=7,
        providers=("aws_s3", "gcs"),
        storage_location="s3://emp-backups/timescale",
        restore_test_interval_days=14,
    )
    backup_state = BackupState(
        last_backup_at=moment - timedelta(hours=2, minutes=30),
        last_backup_status="ok",
        last_restore_test_at=moment - timedelta(days=7),
        last_restore_status="ok",
        recorded_failures=tuple(),
    )

    metadata = {
        "drill": {
            "initiated_at": moment.isoformat(),
            "scenario": scenario,
        }
    }

    return simulate_disaster_recovery_drill(
        plan=plan,
        ingest_results=ingest_results,
        fail_dimensions=fail_dimensions,
        backup_policy=backup_policy,
        backup_state=backup_state,
        scenario=scenario,
        now=moment,
        metadata=metadata,
    )


__all__ = [
    "DisasterRecoveryReport",
    "DisasterRecoveryStatus",
    "RecoveryStep",
    "format_disaster_recovery_markdown",
    "plan_disaster_recovery",
    "simulate_default_disaster_recovery",
    "simulate_disaster_recovery_drill",
]
