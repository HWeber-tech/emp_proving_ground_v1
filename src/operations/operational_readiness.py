"""Aggregate operational readiness telemetry across incident and validation feeds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, Sequence

from src.core.event_bus import Event, EventBus
from src.operations.alerts import (
    AlertDispatchResult,
    AlertEvent,
    AlertManager,
    AlertSeverity,
)
from src.operations.event_bus_failover import publish_event_with_failover
from src.operations.incident_response import (
    IncidentResponseMetrics,
    IncidentResponseSnapshot,
    IncidentResponseStatus,
)
from src.operations.drift_sentry import DriftSentryMetric, DriftSentrySnapshot
from src.operations.sensory_drift import DriftSeverity, SensoryDriftSnapshot
from src.operations.slo import OperationalSLOSnapshot, SLOStatus
from src.operations.system_validation import (
    SystemValidationSnapshot,
    SystemValidationStatus,
)


logger = logging.getLogger(__name__)


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


def _map_drift_status(severity: DriftSeverity) -> OperationalReadinessStatus:
    if severity is DriftSeverity.alert:
        return OperationalReadinessStatus.fail
    if severity is DriftSeverity.warn:
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
    metrics = snapshot.metrics
    if metrics is None:
        metrics_payload = snapshot.metadata.get("reliability_metrics")
        if isinstance(metrics_payload, Mapping):
            metrics = IncidentResponseMetrics.from_mapping({
                "INCIDENT_METRICS_MTTA_MINUTES": metrics_payload.get("mtta_minutes"),
                "INCIDENT_METRICS_MTTR_MINUTES": metrics_payload.get("mttr_minutes"),
            })
    if metrics is not None:
        if metrics.mtta_minutes is not None:
            summary_parts.append(f"MTTA={metrics.mtta_minutes:.1f}m")
        if metrics.mttr_minutes is not None:
            summary_parts.append(f"MTTR={metrics.mttr_minutes:.1f}m")
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


def _build_metric_detail(metric: DriftSentryMetric) -> dict[str, object]:
    detail: dict[str, object] = {
        "severity": metric.severity.value,
        "baseline_mean": metric.baseline_mean,
        "evaluation_mean": metric.evaluation_mean,
        "baseline_variance": metric.baseline_variance,
        "evaluation_variance": metric.evaluation_variance,
        "baseline_count": metric.baseline_count,
        "evaluation_count": metric.evaluation_count,
    }
    if metric.detectors:
        detail["detectors"] = list(metric.detectors)
    if metric.page_hinkley_stat is not None:
        detail["page_hinkley_stat"] = metric.page_hinkley_stat
    if metric.variance_ratio is not None:
        detail["variance_ratio"] = metric.variance_ratio
    return detail


def _drift_component(
    snapshot: SensoryDriftSnapshot | DriftSentrySnapshot,
) -> OperationalReadinessComponent:
    status = _map_drift_status(snapshot.status)
    degraded: list[str] = []
    issue_counts: dict[str, int] = {}
    issue_details: dict[str, dict[str, object]] = {}

    if isinstance(snapshot, DriftSentrySnapshot):
        for name, metric in snapshot.metrics.items():
            severity = metric.severity
            if severity is DriftSeverity.normal:
                continue
            if severity is DriftSeverity.warn:
                issue_counts["warn"] = issue_counts.get("warn", 0) + 1
            elif severity is DriftSeverity.alert:
                issue_counts["fail"] = issue_counts.get("fail", 0) + 1

            detector_suffix = f" ({', '.join(metric.detectors)})" if metric.detectors else ""
            degraded.append(f"{name}:{severity.value}{detector_suffix}")
            issue_details[name] = _build_metric_detail(metric)

        metadata: dict[str, object] = {
            "snapshot": snapshot.as_dict(),
            "config": snapshot.config.as_dict(),
            "runbook": snapshot.metadata.get(
                "runbook", "docs/operations/runbooks/drift_sentry_response.md"
            ),
        }
    else:
        metadata = {"snapshot": snapshot.as_dict()}
        for name, dimension in snapshot.dimensions.items():
            severity = dimension.severity
            if severity is DriftSeverity.normal:
                continue
            if severity is DriftSeverity.warn:
                issue_counts["warn"] = issue_counts.get("warn", 0) + 1
            elif severity is DriftSeverity.alert:
                issue_counts["fail"] = issue_counts.get("fail", 0) + 1

            detector_suffix = f" ({', '.join(dimension.detectors)})" if dimension.detectors else ""
            degraded.append(f"{name}:{severity.value}{detector_suffix}")

            detail: dict[str, object] = {"severity": severity.value}
            if dimension.detectors:
                detail["detectors"] = list(dimension.detectors)
            if dimension.page_hinkley_stat is not None:
                detail["page_hinkley_stat"] = dimension.page_hinkley_stat
            if dimension.variance_ratio is not None:
                detail["variance_ratio"] = dimension.variance_ratio
            issue_details[name] = detail

    if issue_counts:
        metadata["issue_counts"] = issue_counts
    if issue_details:
        metadata["issue_details"] = issue_details

    summary = (
        "no drift exceedances"
        if not degraded
        else "; ".join(sorted(degraded))
    )

    return OperationalReadinessComponent(
        name="drift_sentry",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _capture_component_issues(
    component: OperationalReadinessComponent,
    aggregated_issue_counts: dict[str, int],
    component_issue_details: dict[str, dict[str, object]],
) -> None:
    metadata = component.metadata if isinstance(component.metadata, Mapping) else None
    if not metadata:
        return

    detail_payload: dict[str, object] = {}

    issue_counts = metadata.get("issue_counts")
    if isinstance(issue_counts, Mapping):
        counts_copy: dict[str, int] = {}
        for raw_severity, raw_count in issue_counts.items():
            severity = str(raw_severity)
            try:
                count_value = int(raw_count)
            except (TypeError, ValueError):
                continue
            aggregated_issue_counts[severity] = (
                aggregated_issue_counts.get(severity, 0) + count_value
            )
            counts_copy[severity] = count_value
        if counts_copy:
            detail_payload["issue_counts"] = counts_copy

    for key in ("issue_details", "issue_catalog", "issue_category_severity", "highest_issue_severity"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            detail_payload[key] = dict(value)
        elif isinstance(value, (list, tuple)):
            detail_payload[key] = list(value)
        else:
            detail_payload[key] = value

    if detail_payload:
        component_issue_details[component.name] = detail_payload


def evaluate_operational_readiness(
    *,
    system_validation: SystemValidationSnapshot | None = None,
    incident_response: IncidentResponseSnapshot | None = None,
    drift_snapshot: SensoryDriftSnapshot | DriftSentrySnapshot | None = None,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OperationalReadinessSnapshot:
    """Aggregate the available readiness telemetry into a single snapshot."""

    components: list[OperationalReadinessComponent] = []
    moments: list[datetime] = []
    overall_status = OperationalReadinessStatus.ok
    aggregated_issue_counts: dict[str, int] = {}
    component_issue_details: dict[str, dict[str, object]] = {}

    if system_validation is not None:
        component = _system_validation_component(system_validation)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(system_validation.generated_at)
        _capture_component_issues(component, aggregated_issue_counts, component_issue_details)

    if incident_response is not None:
        component = _incident_response_component(incident_response)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(incident_response.generated_at)
        _capture_component_issues(component, aggregated_issue_counts, component_issue_details)

    if drift_snapshot is not None:
        component = _drift_component(drift_snapshot)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        drift_generated_at = drift_snapshot.generated_at
        if drift_generated_at.tzinfo is None:
            drift_generated_at = drift_generated_at.replace(tzinfo=UTC)
        moments.append(drift_generated_at)
        _capture_component_issues(component, aggregated_issue_counts, component_issue_details)

    if slo_snapshot is not None:
        component = _slo_component(slo_snapshot)
        components.append(component)
        overall_status = _escalate(overall_status, component.status)
        moments.append(slo_snapshot.generated_at)
        _capture_component_issues(component, aggregated_issue_counts, component_issue_details)

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
    fallback_issue_counts: dict[str, int] = {}
    for component in components:
        if component.status is OperationalReadinessStatus.warn:
            fallback_issue_counts["warn"] = fallback_issue_counts.get("warn", 0) + 1
        elif component.status is OperationalReadinessStatus.fail:
            fallback_issue_counts["fail"] = fallback_issue_counts.get("fail", 0) + 1
        if component.status is not OperationalReadinessStatus.ok:
            detail = component_issue_details.setdefault(
                component.name,
                {
                    "severity": component.status.value,
                    "summary": component.summary,
                },
            )
            counts = detail.setdefault("issue_counts", {})
            if isinstance(counts, dict):
                counts.setdefault(component.status.value, 1)
    if aggregated_issue_counts:
        snapshot_metadata["issue_counts"] = dict(aggregated_issue_counts)
    elif fallback_issue_counts:
        snapshot_metadata["issue_counts"] = fallback_issue_counts
    if component_issue_details:
        snapshot_metadata["component_issue_details"] = component_issue_details
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
    base_tags: Sequence[str] = ("operational-readiness",),
) -> list[AlertEvent]:
    """Translate the readiness snapshot into alert events for routing policies."""

    events: list[AlertEvent] = []
    severity_map: Mapping[OperationalReadinessStatus, AlertSeverity] = {
        OperationalReadinessStatus.ok: AlertSeverity.info,
        OperationalReadinessStatus.warn: AlertSeverity.warning,
        OperationalReadinessStatus.fail: AlertSeverity.critical,
    }

    base_tag_tuple = tuple(base_tags)

    if include_overall and _should_alert(snapshot.status, threshold):
        events.append(
            AlertEvent(
                category="operational.readiness",
                severity=severity_map[snapshot.status],
                message=f"Operational readiness status {snapshot.status.value}",
                tags=base_tag_tuple,
                context={"snapshot": snapshot.as_dict()},
            )
        )

    for component in snapshot.components:
        if not _should_alert(component.status, threshold):
            continue
        component_tags = base_tag_tuple + (component.name,)
        events.append(
            AlertEvent(
                category=f"operational.{component.name}",
                severity=severity_map[component.status],
                message=f"{component.name} status {component.status.value}: {component.summary}",
                tags=component_tags,
                context={"component": component.as_dict(), "snapshot": snapshot.as_dict()},
            )
        )

    return events


def route_operational_readiness_alerts(
    manager: AlertManager,
    snapshot: OperationalReadinessSnapshot,
    *,
    threshold: OperationalReadinessStatus = OperationalReadinessStatus.warn,
    include_overall: bool = True,
    base_tags: Sequence[str] = ("operational-readiness",),
) -> list[AlertDispatchResult]:
    """Dispatch operational readiness alerts through an :class:`AlertManager`."""

    events = derive_operational_alerts(
        snapshot,
        threshold=threshold,
        include_overall=include_overall,
        base_tags=base_tags,
    )
    results: list[AlertDispatchResult] = []
    for event in events:
        results.append(manager.dispatch(event))
    return results


def publish_operational_readiness_snapshot(
    event_bus: EventBus,
    snapshot: OperationalReadinessSnapshot,
    *,
    source: str = "operations.operational_readiness",
) -> None:
    """Publish the operational readiness snapshot onto the runtime event bus."""

    event = Event(
        type="telemetry.operational.operational_readiness",
        payload=snapshot.as_dict(),
        source=source,
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=(
            "Runtime event bus rejected operational readiness snapshot; falling back to global bus"
        ),
        runtime_unexpected_message=(
            "Unexpected error publishing operational readiness snapshot via runtime bus"
        ),
        runtime_none_message=(
            "Runtime event bus returned None while publishing operational readiness snapshot; using global bus"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing operational readiness snapshot"
        ),
        global_unexpected_message=(
            "Unexpected error publishing operational readiness snapshot via global bus"
        ),
    )


__all__ = [
    "OperationalReadinessComponent",
    "OperationalReadinessSnapshot",
    "OperationalReadinessStatus",
    "derive_operational_alerts",
    "evaluate_operational_readiness",
    "format_operational_readiness_markdown",
    "publish_operational_readiness_snapshot",
    "route_operational_readiness_alerts",
]
