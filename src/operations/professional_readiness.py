"""Professional readiness aggregation utilities.

This module fuses the readiness surfaces introduced for the institutional data
backbone into a single snapshot so operators inherit one authoritative view of
professional-tier posture.  It bridges ingest telemetry, backup status,
operational SLO grading, and failover/recovery context into a compact payload
that can be published on the runtime event bus and rendered inside runtime
summaries, mirroring the roadmap's call for concept-aligned operational
readiness signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Callable, Mapping

from src.core.event_bus import Event, EventBus, TopicBus

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.operations.operational_readiness import (
    OperationalReadinessSnapshot,
    OperationalReadinessStatus,
)
from src.operations.slo import OperationalSLOSnapshot, SLOStatus
from src.operations.event_bus_failover import publish_event_with_failover


logger = logging.getLogger(__name__)


class ProfessionalReadinessStatus(Enum):
    """Severity grading for the aggregated readiness snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"

    @classmethod
    def from_backbone(cls, status: BackboneStatus) -> "ProfessionalReadinessStatus":
        if status is BackboneStatus.ok:
            return cls.ok
        if status is BackboneStatus.warn:
            return cls.warn
        return cls.fail

    @classmethod
    def from_backup(cls, status: BackupStatus) -> "ProfessionalReadinessStatus":
        if status is BackupStatus.ok:
            return cls.ok
        if status is BackupStatus.warn:
            return cls.warn
        return cls.fail

    @classmethod
    def from_slo(cls, status: SLOStatus) -> "ProfessionalReadinessStatus":
        if status is SLOStatus.met:
            return cls.ok
        if status is SLOStatus.at_risk:
            return cls.warn
        return cls.fail

    @classmethod
    def from_operational(
        cls, status: OperationalReadinessStatus
    ) -> "ProfessionalReadinessStatus":
        if status is OperationalReadinessStatus.ok:
            return cls.ok
        if status is OperationalReadinessStatus.warn:
            return cls.warn
        return cls.fail


_STATUS_ORDER: Mapping[ProfessionalReadinessStatus, int] = {
    ProfessionalReadinessStatus.ok: 0,
    ProfessionalReadinessStatus.warn: 1,
    ProfessionalReadinessStatus.fail: 2,
}


def _escalate(
    current: ProfessionalReadinessStatus,
    candidate: ProfessionalReadinessStatus,
) -> ProfessionalReadinessStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_moment(moment: datetime) -> datetime:
    """Return a timezone-aware UTC timestamp for aggregation."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


@dataclass(frozen=True)
class ProfessionalReadinessComponent:
    """Structured view of an individual readiness contributor."""

    name: str
    status: ProfessionalReadinessStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ProfessionalReadinessSnapshot:
    """Aggregated readiness snapshot for professional deployments."""

    status: ProfessionalReadinessStatus
    generated_at: datetime
    components: tuple[ProfessionalReadinessComponent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "components": [component.as_dict() for component in self.components],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.components:
            return "| Component | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            rows.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        if self.metadata:
            rows.append("")
            rows.append("Metadata:")
            for key, value in sorted(self.metadata.items()):
                rows.append(f"- **{key}**: {value}")
        return "\n".join(rows)


def _component_from_backbone(
    snapshot: DataBackboneReadinessSnapshot,
) -> ProfessionalReadinessComponent:
    return ProfessionalReadinessComponent(
        name="data_backbone",
        status=ProfessionalReadinessStatus.from_backbone(snapshot.status),
        summary=f"status={snapshot.status.value} across {len(snapshot.components)} components",
        metadata={"snapshot": snapshot.as_dict()},
    )


def _component_from_backup(
    snapshot: BackupReadinessSnapshot,
) -> ProfessionalReadinessComponent:
    return ProfessionalReadinessComponent(
        name="backups",
        status=ProfessionalReadinessStatus.from_backup(snapshot.status),
        summary=f"{snapshot.service} status={snapshot.status.value}",
        metadata={"snapshot": snapshot.as_dict()},
    )


def _component_from_slos(
    snapshot: OperationalSLOSnapshot,
) -> ProfessionalReadinessComponent:
    return ProfessionalReadinessComponent(
        name="ingest_slos",
        status=ProfessionalReadinessStatus.from_slo(snapshot.status),
        summary=f"status={snapshot.status.value} across {len(snapshot.slos)} SLOs",
        metadata={"snapshot": snapshot.as_dict()},
    )


def _component_from_operational_readiness(
    snapshot: OperationalReadinessSnapshot,
) -> ProfessionalReadinessComponent:
    status = ProfessionalReadinessStatus.from_operational(snapshot.status)
    summary = (
        f"status={snapshot.status.value} across {len(snapshot.components)} components"
    )
    metadata: dict[str, object] = {"snapshot": snapshot.as_dict()}
    snapshot_metadata = snapshot.metadata
    if isinstance(snapshot_metadata, Mapping):
        breakdown = snapshot_metadata.get("status_breakdown")
        if isinstance(breakdown, Mapping):
            metadata["status_breakdown"] = dict(breakdown)
        component_statuses = snapshot_metadata.get("component_statuses")
        if isinstance(component_statuses, Mapping):
            metadata["component_statuses"] = dict(component_statuses)
        issue_counts = snapshot_metadata.get("issue_counts")
        if isinstance(issue_counts, Mapping):
            metadata["issue_counts"] = {
                str(severity): int(count)
                for severity, count in issue_counts.items()
                if isinstance(count, int) or isinstance(count, float)
            }
        issue_details = snapshot_metadata.get("component_issue_details")
        if isinstance(issue_details, Mapping):
            metadata["component_issue_details"] = {
                name: dict(detail)
                for name, detail in issue_details.items()
                if isinstance(detail, Mapping)
            }
    return ProfessionalReadinessComponent(
        name="operational_readiness",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _component_from_failover(
    decision: IngestFailoverDecision,
) -> ProfessionalReadinessComponent:
    if decision.should_failover:
        status = ProfessionalReadinessStatus.fail
        summary = decision.reason or "failover triggered"
    elif decision.optional_triggers:
        status = ProfessionalReadinessStatus.warn
        summary = "optional dimensions degraded"
    else:
        status = ProfessionalReadinessStatus.ok
        summary = "no failover required"

    return ProfessionalReadinessComponent(
        name="failover",
        status=status,
        summary=summary,
        metadata=decision.as_dict(),
    )


def _component_from_recovery(
    recommendation: IngestRecoveryRecommendation,
) -> ProfessionalReadinessComponent:
    if recommendation.is_empty():
        status = ProfessionalReadinessStatus.ok
        summary = "no recovery actions necessary"
    else:
        status = ProfessionalReadinessStatus.warn
        summary = "recovery actions planned"

    metadata = recommendation.summary()
    return ProfessionalReadinessComponent(
        name="recovery",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def evaluate_professional_readiness(
    *,
    backbone_snapshot: DataBackboneReadinessSnapshot | None = None,
    backup_snapshot: BackupReadinessSnapshot | None = None,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    operational_readiness_snapshot: OperationalReadinessSnapshot | None = None,
    failover_decision: IngestFailoverDecision | None = None,
    recovery_recommendation: IngestRecoveryRecommendation | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ProfessionalReadinessSnapshot:
    """Aggregate institutional readiness signals into a professional snapshot."""

    components: list[ProfessionalReadinessComponent] = []
    overall = ProfessionalReadinessStatus.ok
    moments: list[datetime] = []

    if backbone_snapshot is not None:
        component = _component_from_backbone(backbone_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)
        moments.append(_normalise_moment(backbone_snapshot.generated_at))

    if backup_snapshot is not None:
        component = _component_from_backup(backup_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)
        moments.append(_normalise_moment(backup_snapshot.generated_at))

    if slo_snapshot is not None:
        component = _component_from_slos(slo_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)
        moments.append(_normalise_moment(slo_snapshot.generated_at))

    if operational_readiness_snapshot is not None:
        component = _component_from_operational_readiness(operational_readiness_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)
        moments.append(_normalise_moment(operational_readiness_snapshot.generated_at))

    if failover_decision is not None:
        component = _component_from_failover(failover_decision)
        components.append(component)
        overall = _escalate(overall, component.status)
        moments.append(_normalise_moment(failover_decision.generated_at))

    if recovery_recommendation is not None:
        component = _component_from_recovery(recovery_recommendation)
        components.append(component)
        overall = _escalate(overall, component.status)

    generated_at = max(moments) if moments else datetime.now(tz=UTC)

    status_breakdown: dict[str, int] = {}
    component_statuses: dict[str, str] = {}
    aggregated_issue_counts: dict[str, int] = {}
    component_issue_details: dict[str, dict[str, object]] = {}

    for component in components:
        status_value = component.status.value
        status_breakdown[status_value] = status_breakdown.get(status_value, 0) + 1
        component_statuses[component.name] = status_value

        metadata_payload = component.metadata
        if isinstance(metadata_payload, Mapping):
            processed_counts = False
            issue_counts = metadata_payload.get("issue_counts")
            if isinstance(issue_counts, Mapping):
                normalised_counts: dict[str, int] = {}
                for severity, count in issue_counts.items():
                    try:
                        normalised_counts[str(severity)] = int(count)
                    except (TypeError, ValueError):
                        continue
                if normalised_counts:
                    for severity, count in normalised_counts.items():
                        aggregated_issue_counts[severity] = (
                            aggregated_issue_counts.get(severity, 0) + count
                        )
                    component_issue_details.setdefault(component.name, {})[
                        "issue_counts"
                    ] = normalised_counts
                    processed_counts = True

            detail_payload = metadata_payload.get("component_issue_details")
            if isinstance(detail_payload, Mapping) and detail_payload:
                component_issue_details.setdefault(component.name, {})[
                    "component_issue_details"
                ] = {
                    key: dict(value)
                    for key, value in detail_payload.items()
                    if isinstance(value, Mapping)
                }

            snapshot_payload = metadata_payload.get("snapshot")
            if isinstance(snapshot_payload, Mapping) and not processed_counts:
                nested_metadata = snapshot_payload.get("metadata")
                if isinstance(nested_metadata, Mapping):
                    nested_counts = nested_metadata.get("issue_counts")
                    if isinstance(nested_counts, Mapping):
                        normalised_counts: dict[str, int] = {}
                        for severity, count in nested_counts.items():
                            try:
                                normalised_counts[str(severity)] = int(count)
                            except (TypeError, ValueError):
                                continue
                        if normalised_counts:
                            for severity, count in normalised_counts.items():
                                aggregated_issue_counts[severity] = (
                                    aggregated_issue_counts.get(severity, 0) + count
                                )
                            component_issue_details.setdefault(component.name, {})[
                                "snapshot_issue_counts"
                            ] = normalised_counts

    snapshot_metadata: dict[str, object] = {
        "component_count": len(components),
    }
    if status_breakdown:
        snapshot_metadata["status_breakdown"] = status_breakdown
    if component_statuses:
        snapshot_metadata["component_statuses"] = component_statuses
    if aggregated_issue_counts:
        snapshot_metadata["issue_counts"] = aggregated_issue_counts
    if component_issue_details:
        snapshot_metadata["component_issue_details"] = component_issue_details
    if metadata:
        snapshot_metadata.update(dict(metadata))

    return ProfessionalReadinessSnapshot(
        status=overall,
        generated_at=generated_at,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_professional_readiness_markdown(
    snapshot: ProfessionalReadinessSnapshot,
) -> str:
    """Render a professional readiness snapshot as Markdown."""

    return snapshot.to_markdown()


def publish_professional_readiness_snapshot(
    event_bus: EventBus,
    snapshot: ProfessionalReadinessSnapshot,
    *,
    source: str = "operations.professional_readiness",
    global_bus_factory: Callable[[], TopicBus] | None = None,
) -> None:
    """Publish the snapshot onto the runtime/global event buses."""

    event = Event(
        type="telemetry.operational.readiness",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Primary event bus publish_from_sync failed; falling back to global bus "
            "for professional readiness telemetry"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing professional readiness snapshot via runtime bus"
        ),
        runtime_none_message=(
            "Primary event bus publish_from_sync returned None; falling back to global bus "
            "for professional readiness telemetry"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing professional readiness snapshot"
        ),
        global_unexpected_message=(
            "Unexpected error publishing professional readiness snapshot via global bus"
        ),
        global_bus_factory=global_bus_factory,
    )


__all__ = [
    "ProfessionalReadinessComponent",
    "ProfessionalReadinessSnapshot",
    "ProfessionalReadinessStatus",
    "format_professional_readiness_markdown",
    "publish_professional_readiness_snapshot",
    "evaluate_professional_readiness",
]
