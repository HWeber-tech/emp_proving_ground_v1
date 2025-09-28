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
from typing import Mapping

from src.core.event_bus import Event, EventBus, get_global_bus

from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.operations.slo import OperationalSLOSnapshot, SLOStatus


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
    failover_decision: IngestFailoverDecision | None = None,
    recovery_recommendation: IngestRecoveryRecommendation | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ProfessionalReadinessSnapshot:
    """Aggregate institutional readiness signals into a professional snapshot."""

    generated_at = datetime.now(tz=UTC)
    components: list[ProfessionalReadinessComponent] = []
    overall = ProfessionalReadinessStatus.ok

    if backbone_snapshot is not None:
        component = _component_from_backbone(backbone_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)

    if backup_snapshot is not None:
        component = _component_from_backup(backup_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)

    if slo_snapshot is not None:
        component = _component_from_slos(slo_snapshot)
        components.append(component)
        overall = _escalate(overall, component.status)

    if failover_decision is not None:
        component = _component_from_failover(failover_decision)
        components.append(component)
        overall = _escalate(overall, component.status)

    if recovery_recommendation is not None:
        component = _component_from_recovery(recovery_recommendation)
        components.append(component)
        overall = _escalate(overall, component.status)

    snapshot_metadata = dict(metadata or {})
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
) -> None:
    """Publish the snapshot onto the runtime/global event buses."""

    event = Event(
        type="telemetry.operational.readiness",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish professional readiness via runtime event bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug(
            "Professional readiness telemetry publish skipped", exc_info=True
        )


__all__ = [
    "ProfessionalReadinessComponent",
    "ProfessionalReadinessSnapshot",
    "ProfessionalReadinessStatus",
    "format_professional_readiness_markdown",
    "publish_professional_readiness_snapshot",
    "evaluate_professional_readiness",
]
