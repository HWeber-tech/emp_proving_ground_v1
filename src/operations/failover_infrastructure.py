"""Aggregates failover infrastructure readiness across operational telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, Sequence

from src.operations.backup import (
    BackupReadinessSnapshot,
    BackupStatus,
)
from src.operations.cross_region_failover import (
    CrossRegionFailoverSnapshot,
    CrossRegionStatus,
)
from src.operations.failover_drill import (
    FailoverDrillSnapshot,
    FailoverDrillStatus,
)


class FailoverInfrastructureStatus(StrEnum):
    """High-level readiness classification for failover infrastructure."""

    ready = "ready"
    degraded = "degraded"
    blocked = "blocked"


_STATUS_ORDER: Mapping[FailoverInfrastructureStatus, int] = {
    FailoverInfrastructureStatus.ready: 0,
    FailoverInfrastructureStatus.degraded: 1,
    FailoverInfrastructureStatus.blocked: 2,
}

_BACKUP_TO_INFRA: Mapping[BackupStatus, FailoverInfrastructureStatus] = {
    BackupStatus.ok: FailoverInfrastructureStatus.ready,
    BackupStatus.warn: FailoverInfrastructureStatus.degraded,
    BackupStatus.fail: FailoverInfrastructureStatus.blocked,
}

_FAILOVER_TO_INFRA: Mapping[FailoverDrillStatus, FailoverInfrastructureStatus] = {
    FailoverDrillStatus.ok: FailoverInfrastructureStatus.ready,
    FailoverDrillStatus.warn: FailoverInfrastructureStatus.degraded,
    FailoverDrillStatus.fail: FailoverInfrastructureStatus.blocked,
}

_CROSS_REGION_TO_INFRA: Mapping[CrossRegionStatus, FailoverInfrastructureStatus] = {
    CrossRegionStatus.ok: FailoverInfrastructureStatus.ready,
    CrossRegionStatus.warn: FailoverInfrastructureStatus.degraded,
    CrossRegionStatus.fail: FailoverInfrastructureStatus.blocked,
}


def _combine_status(
    current: FailoverInfrastructureStatus,
    candidate: FailoverInfrastructureStatus,
) -> FailoverInfrastructureStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class FailoverResource:
    """Declarative description of a service protected by failover."""

    name: str
    primary_location: str
    secondary_location: str
    tier: str
    rto_seconds: float
    rpo_seconds: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "primary_location": self.primary_location,
            "secondary_location": self.secondary_location,
            "tier": self.tier,
            "rto_seconds": self.rto_seconds,
            "rpo_seconds": self.rpo_seconds,
            "notes": list(self.notes),
        }
        return payload


@dataclass(frozen=True)
class FailoverInfrastructureComponent:
    """Individual component contributing to the failover posture."""

    name: str
    status: FailoverInfrastructureStatus
    summary: str
    details: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "details": list(self.details),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class FailoverInfrastructurePlan:
    """Aggregated readiness view across backups, drills, and replication."""

    status: FailoverInfrastructureStatus
    generated_at: datetime
    components: tuple[FailoverInfrastructureComponent, ...]
    resources: tuple[FailoverResource, ...]
    backup_snapshot: BackupReadinessSnapshot | None = None
    failover_snapshot: FailoverDrillSnapshot | None = None
    cross_region_snapshot: CrossRegionFailoverSnapshot | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "components": [component.as_dict() for component in self.components],
            "resources": [resource.as_dict() for resource in self.resources],
            "metadata": dict(self.metadata),
        }
        if self.backup_snapshot is not None:
            payload["backup_snapshot"] = self.backup_snapshot.as_dict()
        if self.failover_snapshot is not None:
            payload["failover_snapshot"] = self.failover_snapshot.as_dict()
        if self.cross_region_snapshot is not None:
            payload["cross_region_snapshot"] = self.cross_region_snapshot.as_dict()
        return payload

    def to_markdown(self) -> str:
        lines = [
            "# Failover infrastructure readiness",
            "",
            f"- Status: {self.status.value.upper()}",
            f"- Generated: {self.generated_at.astimezone(UTC).isoformat()}",
            "",
        ]

        if self.resources:
            lines.append("## Protected resources")
            lines.append("| Service | Tier | Primary | Secondary | RTO (s) | RPO (s) |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for resource in self.resources:
                lines.append(
                    "| {name} | {tier} | {primary} | {secondary} | {rto:.0f} | {rpo:.0f} |".format(
                        name=resource.name,
                        tier=resource.tier,
                        primary=resource.primary_location,
                        secondary=resource.secondary_location,
                        rto=resource.rto_seconds,
                        rpo=resource.rpo_seconds,
                    )
                )
            lines.append("")
            for resource in self.resources:
                if not resource.notes:
                    continue
                lines.append(f"### {resource.name} notes")
                for note in resource.notes:
                    lines.append(f"- {note}")
                lines.append("")

        if self.components:
            lines.append("## Components")
            lines.append("| Component | Status | Summary |")
            lines.append("| --- | --- | --- |")
            for component in self.components:
                lines.append(
                    f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
                )
            lines.append("")

        if any(component.details for component in self.components):
            lines.append("## Component details")
            for component in self.components:
                if not component.details:
                    continue
                lines.append(f"### {component.name}")
                for detail in component.details:
                    lines.append(f"- {detail}")
                lines.append("")

        if self.backup_snapshot is not None:
            lines.append("## Backup readiness snapshot")
            lines.append(self.backup_snapshot.to_markdown())
            lines.append("")

        if self.failover_snapshot is not None:
            lines.append("## Failover drill snapshot")
            lines.append(self.failover_snapshot.to_markdown())
            lines.append("")

        if self.cross_region_snapshot is not None:
            lines.append("## Cross-region replication snapshot")
            lines.append(self.cross_region_snapshot.to_markdown())

        return "\n".join(lines).rstrip()


def format_failover_infrastructure_markdown(plan: FailoverInfrastructurePlan) -> str:
    """Consistency helper mirroring other operational formatters."""

    return plan.to_markdown()


def _build_resource_component(
    resources: Sequence[FailoverResource],
) -> FailoverInfrastructureComponent:
    if not resources:
        return FailoverInfrastructureComponent(
            name="Resource coverage",
            status=FailoverInfrastructureStatus.blocked,
            summary="No failover resources defined",
            details=("Define primary/secondary mappings before go-live",),
        )

    details = [
        f"{resource.name}: {resource.tier} tier – {resource.primary_location} → {resource.secondary_location}"
        for resource in resources
    ]
    return FailoverInfrastructureComponent(
        name="Resource coverage",
        status=FailoverInfrastructureStatus.ready,
        summary=f"{len(resources)} resources protected",
        details=tuple(details),
    )


def _build_backup_component(
    snapshot: BackupReadinessSnapshot | None,
) -> FailoverInfrastructureComponent:
    if snapshot is None:
        return FailoverInfrastructureComponent(
            name="Backups",
            status=FailoverInfrastructureStatus.degraded,
            summary="Backup snapshot unavailable",
            details=("Ensure backup telemetry is connected",),
        )

    status = _BACKUP_TO_INFRA[snapshot.status]
    details = [
        f"Retention: {snapshot.retention_days} days",
    ]
    if snapshot.latest_backup_at is not None:
        details.append(f"Latest backup: {snapshot.latest_backup_at.astimezone(UTC).isoformat()}")
    else:
        details.append("Latest backup: not recorded")
    if snapshot.next_backup_due_at is not None:
        details.append(
            f"Next backup due: {snapshot.next_backup_due_at.astimezone(UTC).isoformat()}"
        )
    if snapshot.issues:
        details.extend(snapshot.issues)

    return FailoverInfrastructureComponent(
        name="Backups",
        status=status,
        summary=f"Backup status {snapshot.status.value}",
        details=tuple(details),
        metadata={"service": snapshot.service},
    )


def _build_failover_component(
    snapshot: FailoverDrillSnapshot | None,
) -> FailoverInfrastructureComponent:
    if snapshot is None:
        return FailoverInfrastructureComponent(
            name="Failover automation",
            status=FailoverInfrastructureStatus.degraded,
            summary="No failover drill available",
            details=("Execute failover drill to validate runbooks",),
        )

    status = _FAILOVER_TO_INFRA[snapshot.status]
    details = [
        f"Scenario: {snapshot.scenario}",
    ]
    for component in snapshot.components:
        details.append(
            f"{component.name}: {component.status.value} – {component.summary}"
        )

    fallback = snapshot.metadata.get("fallback") if isinstance(snapshot.metadata, Mapping) else None
    if isinstance(fallback, Mapping):
        executed = fallback.get("executed")
        error = fallback.get("error")
        if executed:
            details.append("Fallback executed during drill")
        if error:
            details.append(f"Fallback error: {error}")

    return FailoverInfrastructureComponent(
        name="Failover automation",
        status=status,
        summary=f"Failover status {snapshot.status.value}",
        details=tuple(details),
    )


def _build_cross_region_component(
    snapshot: CrossRegionFailoverSnapshot | None,
) -> FailoverInfrastructureComponent:
    if snapshot is None:
        return FailoverInfrastructureComponent(
            name="Cross-region replication",
            status=FailoverInfrastructureStatus.degraded,
            summary="No cross-region snapshot captured",
            details=("Run replication telemetry to confirm posture",),
        )

    status = _CROSS_REGION_TO_INFRA[snapshot.status]
    details = [
        f"Primary region: {snapshot.primary_region}",
        f"Replica region: {snapshot.replica_region}",
    ]
    for component in snapshot.components:
        details.append(
            f"{component.name}: {component.status.value} – {component.summary}"
        )

    return FailoverInfrastructureComponent(
        name="Cross-region replication",
        status=status,
        summary=f"Cross-region status {snapshot.status.value}",
        details=tuple(details),
    )


def plan_failover_infrastructure(
    resources: Sequence[FailoverResource],
    *,
    backup_snapshot: BackupReadinessSnapshot | None = None,
    failover_snapshot: FailoverDrillSnapshot | None = None,
    cross_region_snapshot: CrossRegionFailoverSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> FailoverInfrastructurePlan:
    """Produce an aggregated failover infrastructure readiness plan."""

    moment = generated_at or datetime.now(tz=UTC)
    resource_component = _build_resource_component(resources)
    backup_component = _build_backup_component(backup_snapshot)
    failover_component = _build_failover_component(failover_snapshot)
    cross_region_component = _build_cross_region_component(cross_region_snapshot)

    components = (
        resource_component,
        backup_component,
        failover_component,
        cross_region_component,
    )

    status = FailoverInfrastructureStatus.ready
    for component in components:
        status = _combine_status(status, component.status)

    plan_metadata = dict(metadata) if metadata else {}
    if resources:
        plan_metadata.setdefault("resource_count", len(resources))
    if backup_snapshot is not None:
        plan_metadata.setdefault("backup_service", backup_snapshot.service)

    return FailoverInfrastructurePlan(
        status=status,
        generated_at=moment,
        components=components,
        resources=tuple(resources),
        backup_snapshot=backup_snapshot,
        failover_snapshot=failover_snapshot,
        cross_region_snapshot=cross_region_snapshot,
        metadata=plan_metadata,
    )
