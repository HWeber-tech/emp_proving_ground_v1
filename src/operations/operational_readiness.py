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


@dataclass(frozen=True)
class OperationalReadinessGateResult:
    """Gate decision for aggregated operational readiness telemetry."""

    status: OperationalReadinessStatus
    should_block: bool
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "should_block": self.should_block,
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

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


def evaluate_operational_readiness_gate(
    snapshot: OperationalReadinessSnapshot,
    *,
    block_on_warn: bool = False,
    fail_components: Sequence[str] = ("system_validation", "incident_response"),
    warn_components: Sequence[str] = (),
    max_fail_components: int | None = 0,
    max_warn_components: int | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> OperationalReadinessGateResult:
    """Assess whether the aggregated readiness posture should block deployments."""

    component_lookup: dict[str, OperationalReadinessComponent] = {
        component.name: component for component in snapshot.components
    }
    component_statuses: dict[str, OperationalReadinessStatus] = {
        name: component.status for name, component in component_lookup.items()
    }
    component_summaries: dict[str, str] = {
        name: component.summary for name, component in component_lookup.items()
    }

    metadata = snapshot.metadata if isinstance(snapshot.metadata, Mapping) else {}
    status_metadata = metadata.get("component_statuses") if isinstance(metadata, Mapping) else None
    if isinstance(status_metadata, Mapping):
        for name, status_value in status_metadata.items():
            if name in component_statuses:
                continue
            try:
                component_statuses[name] = OperationalReadinessStatus(str(status_value))
            except ValueError:
                continue
            summary_catalog = metadata.get("component_issue_details")
            summary_text = ""
            if isinstance(summary_catalog, Mapping):
                detail = summary_catalog.get(name)
                if isinstance(detail, Mapping):
                    summary_text = str(detail.get("summary") or detail.get("headline") or "")
            component_summaries[name] = summary_text

    fail_component_keys = {name.lower() for name in fail_components}
    warn_component_keys = {name.lower() for name in warn_components}

    issue_details = metadata.get("component_issue_details")
    issue_details_map: dict[str, Mapping[str, object]] = {}
    if isinstance(issue_details, Mapping):
        for name, detail in issue_details.items():
            if isinstance(detail, Mapping):
                issue_details_map[name] = detail

    def _component_reason(name: str) -> str:
        status = component_statuses.get(name, snapshot.status)
        summary = component_summaries.get(name)
        detail = issue_details_map.get(name)
        headline = None
        if isinstance(detail, Mapping):
            headline = detail.get("summary") or detail.get("headline")
            if headline:
                headline = str(headline)
        base = f"{name} status {status.value}"
        chosen_summary = headline or summary
        if chosen_summary:
            base += f": {chosen_summary}"
        return base

    gate_status = snapshot.status
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    if snapshot.status is OperationalReadinessStatus.fail:
        blocking_reasons.append("Operational readiness status is FAIL")
        gate_status = OperationalReadinessStatus.fail
    elif snapshot.status is OperationalReadinessStatus.warn:
        reason = "Operational readiness status is WARN"
        if block_on_warn:
            blocking_reasons.append(reason + " and block_on_warn is enabled")
        else:
            warnings.append(reason)
        gate_status = _escalate(gate_status, OperationalReadinessStatus.warn)

    failing_components = [
        name
        for name, status in component_statuses.items()
        if status is OperationalReadinessStatus.fail
    ]
    warning_components = [
        name
        for name, status in component_statuses.items()
        if status is OperationalReadinessStatus.warn
    ]

    for name in failing_components:
        if not fail_component_keys or name.lower() in fail_component_keys:
            blocking_reasons.append(_component_reason(name))
            gate_status = OperationalReadinessStatus.fail

    for name in warning_components:
        reason_text = _component_reason(name)
        if name.lower() in warn_component_keys or block_on_warn:
            blocking_reasons.append(reason_text)
        else:
            warnings.append(reason_text)
        gate_status = _escalate(gate_status, OperationalReadinessStatus.warn)

    if max_fail_components is not None and len(failing_components) > max_fail_components:
        blocking_reasons.append(
            f"{len(failing_components)} components failing exceeds limit {max_fail_components}"
        )
        gate_status = OperationalReadinessStatus.fail

    if max_warn_components is not None and len(warning_components) > max_warn_components:
        message = (
            f"{len(warning_components)} components warning exceeds limit {max_warn_components}"
        )
        if block_on_warn:
            blocking_reasons.append(message)
        else:
            warnings.append(message)
        gate_status = _escalate(gate_status, OperationalReadinessStatus.warn)

    issue_counts = metadata.get("issue_counts") if isinstance(metadata, Mapping) else None
    if isinstance(issue_counts, Mapping) and not blocking_reasons:
        fail_count = int(issue_counts.get("fail", 0))
        if fail_count > 0:
            blocking_reasons.append(
                f"Aggregated issue count reports {fail_count} failing entries"
            )
            gate_status = OperationalReadinessStatus.fail

    gate_metadata: dict[str, object] = {
        "block_on_warn": block_on_warn,
        "fail_components": tuple(fail_components),
        "warn_components": tuple(warn_components),
        "max_fail_components": max_fail_components,
        "max_warn_components": max_warn_components,
        "component_statuses": {
            name: status.value for name, status in component_statuses.items()
        },
    }
    if warning_components:
        gate_metadata["warning_components"] = tuple(warning_components)
    if failing_components:
        gate_metadata["failing_components"] = tuple(failing_components)
    if issue_counts is not None:
        gate_metadata["issue_counts"] = dict(issue_counts)
    if issue_details_map:
        gate_metadata["component_issue_details"] = {
            name: dict(detail) for name, detail in issue_details_map.items()
        }
    if extra_metadata:
        gate_metadata.update(dict(extra_metadata))

    should_block = bool(blocking_reasons)

    return OperationalReadinessGateResult(
        status=gate_status,
        should_block=should_block,
        blocking_reasons=tuple(dict.fromkeys(blocking_reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        metadata=gate_metadata,
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
    include_gate_event: bool = False,
    gate_result: OperationalReadinessGateResult | None = None,
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

    if include_gate_event:
        evaluated_gate = gate_result
        if evaluated_gate is None:
            evaluated_gate = evaluate_operational_readiness_gate(snapshot)
        if evaluated_gate is not None and _should_alert(evaluated_gate.status, threshold):
            if evaluated_gate.blocking_reasons:
                headline = "; ".join(evaluated_gate.blocking_reasons[:2])
            elif evaluated_gate.warnings:
                headline = "; ".join(evaluated_gate.warnings[:2])
            else:
                headline = "no additional context"
            events.append(
                AlertEvent(
                    category="operational.readiness.gate",
                    severity=severity_map[evaluated_gate.status],
                    message=f"Operational readiness gate {evaluated_gate.status.value}: {headline}",
                    tags=base_tag_tuple + ("gate",),
                    context={
                        "snapshot": snapshot.as_dict(),
                        "gate": evaluated_gate.as_dict(),
                    },
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
    include_gate_event: bool = False,
    gate_result: OperationalReadinessGateResult | None = None,
) -> list[AlertDispatchResult]:
    """Dispatch operational readiness alerts through an :class:`AlertManager`."""

    events = derive_operational_alerts(
        snapshot,
        threshold=threshold,
        include_overall=include_overall,
        base_tags=base_tags,
        include_gate_event=include_gate_event,
        gate_result=gate_result,
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
    "OperationalReadinessGateResult",
    "derive_operational_alerts",
    "evaluate_operational_readiness",
    "evaluate_operational_readiness_gate",
    "format_operational_readiness_markdown",
    "publish_operational_readiness_snapshot",
    "route_operational_readiness_alerts",
]
