"""Aggregate operational readiness telemetry across incident and validation feeds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.operations.alerts import AlertEvent, AlertSeverity
from src.operations.incident_response import (
    IncidentResponseSnapshot,
    IncidentResponseStatus,
)
from src.operations.slo import OperationalSLOSnapshot, SLOStatus
from src.operations.system_validation import (
    SystemValidationSnapshot,
    SystemValidationStatus,
)


class OperationalReadinessStatus(StrEnum):
    """Severity grading for the operational readiness surface."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[OperationalReadinessStatus, int] = {
    OperationalReadinessStatus.ok: 0,
    OperationalReadinessStatus.warn: 1,
    OperationalReadinessStatus.fail: 2,
}


def _escalate(
    current: OperationalReadinessStatus, candidate: OperationalReadinessStatus
) -> OperationalReadinessStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _map_system_validation_status(
    status: SystemValidationStatus,
) -> OperationalReadinessStatus:
    if status is SystemValidationStatus.fail:
        return OperationalReadinessStatus.fail
    if status is SystemValidationStatus.warn:
        return OperationalReadinessStatus.warn
    return OperationalReadinessStatus.ok


def _map_incident_status(status: IncidentResponseStatus) -> OperationalReadinessStatus:
    if status is IncidentResponseStatus.fail:
        return OperationalReadinessStatus.fail
    if status is IncidentResponseStatus.warn:
        return OperationalReadinessStatus.warn
    return OperationalReadinessStatus.ok


def _map_slo_status(status: SLOStatus) -> OperationalReadinessStatus:
    if status is SLOStatus.breached:
        return OperationalReadinessStatus.fail
    if status is SLOStatus.at_risk:
        return OperationalReadinessStatus.warn
    return OperationalReadinessStatus.ok


@dataclass(frozen=True)
class OperationalReadinessComponent:
    """Individual contributor to the aggregated readiness posture."""

    name: str
    status: OperationalReadinessStatus
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
class OperationalReadinessSnapshot:
    """Aggregated operational readiness snapshot."""

    status: OperationalReadinessStatus
    generated_at: datetime
    components: tuple[OperationalReadinessComponent, ...]
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
            return (
                f"**Operational readiness** – status: {self.status.value}\n"
                f"Generated at: {self.generated_at.isoformat()}"
            )

        header = (
            f"**Operational readiness** – status: {self.status.value}"
            f"\nGenerated at: {self.generated_at.isoformat()}"
        )
        lines = [header, "", "| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            lines.append(
                f"| {component.name} | {component.status.value} | {component.summary} |"
            )
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)


def _system_validation_component(
    snapshot: SystemValidationSnapshot,
) -> OperationalReadinessComponent:
    status = _map_system_validation_status(snapshot.status)
    failing = [check.name for check in snapshot.checks if not check.passed]
    summary = (
        f"{snapshot.passed_checks}/{snapshot.total_checks} checks passed"
        f" (success={snapshot.success_rate:.0%})"
    )
    if failing:
        summary += "; failing checks: " + ", ".join(sorted(failing))
    metadata = {
        "snapshot": snapshot.as_dict(),
        "failed_checks": failing,
    }
    return OperationalReadinessComponent(
        name="system_validation",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _incident_response_component(
    snapshot: IncidentResponseSnapshot,
) -> OperationalReadinessComponent:
    status = _map_incident_status(snapshot.status)
    summary_parts = [
        f"open incidents={len(snapshot.open_incidents)}",
        f"missing runbooks={len(snapshot.missing_runbooks)}",
    ]
    if snapshot.issues:
        summary_parts.append("issues: " + "; ".join(snapshot.issues))
    summary = ", ".join(summary_parts)
    metadata = {
        "snapshot": snapshot.as_dict(),
    }
    return OperationalReadinessComponent(
        name="incident_response",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _slo_component(snapshot: OperationalSLOSnapshot) -> OperationalReadinessComponent:
    status = _map_slo_status(snapshot.status)
    summary = f"{snapshot.status.value} across {len(snapshot.slos)} SLOs"
    metadata = {"snapshot": snapshot.as_dict()}
    return OperationalReadinessComponent(
        name="operational_slos",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def evaluate_operational_readiness(
    *,
    system_validation: SystemValidationSnapshot | None = None,
    incident_response: IncidentResponseSnapshot | None = None,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OperationalReadinessSnapshot:
    """Aggregate the available readiness telemetry into a single snapshot."""

    components: list[OperationalReadinessComponent] = []
    moments: list[datetime] = []
    overall_status = OperationalReadinessStatus.ok

    if system_validation is not None:
        component = _system_validation_component(system_validation)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(system_validation.generated_at)

    if incident_response is not None:
        component = _incident_response_component(incident_response)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(incident_response.generated_at)

    if slo_snapshot is not None:
        component = _slo_component(slo_snapshot)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(slo_snapshot.generated_at)

    generated_at = max(moments) if moments else datetime.now(tz=UTC)

    snapshot_metadata: dict[str, object] = {"component_count": len(components)}
    if components:
        breakdown: dict[str, int] = {}
        component_statuses: dict[str, str] = {}
        for component in components:
            status_value = component.status.value
            breakdown[status_value] = breakdown.get(status_value, 0) + 1
            component_statuses[component.name] = status_value
        snapshot_metadata["status_breakdown"] = breakdown
        snapshot_metadata["component_statuses"] = component_statuses
    if metadata:
        snapshot_metadata.update(dict(metadata))

    return OperationalReadinessSnapshot(
        status=overall_status,
        generated_at=generated_at,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_operational_readiness_markdown(
    snapshot: OperationalReadinessSnapshot,
) -> str:
    """Convenience wrapper mirroring other operations formatters."""

    return snapshot.to_markdown()


def _should_alert(
    status: OperationalReadinessStatus,
    threshold: OperationalReadinessStatus,
) -> bool:
    return _STATUS_ORDER[status] >= _STATUS_ORDER[threshold]


def derive_operational_alerts(
    snapshot: OperationalReadinessSnapshot,
    *,
    threshold: OperationalReadinessStatus = OperationalReadinessStatus.warn,
    include_overall: bool = True,
) -> list[AlertEvent]:
    """Translate the readiness snapshot into alert events for routing policies."""

    events: list[AlertEvent] = []
    severity_map: Mapping[OperationalReadinessStatus, AlertSeverity] = {
        OperationalReadinessStatus.ok: AlertSeverity.info,
        OperationalReadinessStatus.warn: AlertSeverity.warning,
        OperationalReadinessStatus.fail: AlertSeverity.critical,
    }

    if include_overall and _should_alert(snapshot.status, threshold):
        events.append(
            AlertEvent(
                category="operational.readiness",
                severity=severity_map[snapshot.status],
                message=f"Operational readiness status {snapshot.status.value}",
                context={"snapshot": snapshot.as_dict()},
            )
        )

    for component in snapshot.components:
        if not _should_alert(component.status, threshold):
            continue
        events.append(
            AlertEvent(
                category=f"operational.{component.name}",
                severity=severity_map[component.status],
                message=f"{component.name} status {component.status.value}: {component.summary}",
                context={"component": component.as_dict(), "snapshot": snapshot.as_dict()},
            )
        )

    return events


__all__ = [
    "OperationalReadinessComponent",
    "OperationalReadinessSnapshot",
    "OperationalReadinessStatus",
    "derive_operational_alerts",
    "evaluate_operational_readiness",
    "format_operational_readiness_markdown",
]

