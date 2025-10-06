from __future__ import annotations

"""Textual observability dashboard aligned with the high-impact roadmap."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.operations.event_bus_health import (
    EventBusHealthSnapshot,
    EventBusHealthStatus,
)
from src.operations.operational_readiness import (
    OperationalReadinessSnapshot,
    OperationalReadinessStatus,
)
from src.operations.quality_telemetry import (
    QualityStatus,
    QualityTelemetrySnapshot,
)
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot
from src.operations.slo import OperationalSLOSnapshot, SLOStatus


class DashboardStatus(StrEnum):
    """Severity levels surfaced by the observability dashboard."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[DashboardStatus, int] = {
    DashboardStatus.ok: 0,
    DashboardStatus.warn: 1,
    DashboardStatus.fail: 2,
}


def _escalate(
    current: DashboardStatus, candidate: DashboardStatus
) -> DashboardStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class DashboardPanel:
    """Single panel rendered on the observability dashboard."""

    name: str
    status: DashboardStatus
    headline: str
    details: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "headline": self.headline,
        }
        if self.details:
            payload["details"] = list(self.details)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ObservabilityDashboard:
    """Aggregated dashboard exposing high-impact telemetry slices."""

    generated_at: datetime
    status: DashboardStatus
    panels: Sequence[DashboardPanel]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "panels": [panel.as_dict() for panel in self.panels],
            "metadata": dict(self.metadata),
        }
        payload["remediation_summary"] = self.remediation_summary()
        return payload

    def remediation_summary(self) -> dict[str, Any]:
        """Summarise remediation posture for CI progress tracking."""

        panel_counts = Counter(panel.status for panel in self.panels)
        failing = tuple(
            panel.name for panel in self.panels if panel.status is DashboardStatus.fail
        )
        warnings = tuple(
            panel.name for panel in self.panels if panel.status is DashboardStatus.warn
        )
        healthy = tuple(
            panel.name for panel in self.panels if panel.status is DashboardStatus.ok
        )

        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "overall_status": self.status.value,
            "panel_counts": {
                DashboardStatus.ok.value: panel_counts.get(DashboardStatus.ok, 0),
                DashboardStatus.warn.value: panel_counts.get(DashboardStatus.warn, 0),
                DashboardStatus.fail.value: panel_counts.get(DashboardStatus.fail, 0),
            },
            "failing_panels": failing,
            "warning_panels": warnings,
            "healthy_panels": healthy,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Operational observability dashboard",
            f"- Generated at: {self.generated_at.astimezone(UTC).isoformat()}",
            f"- Overall status: {self.status.value.upper()}",
            "",
        ]

        if self.panels:
            lines.extend(["| Panel | Status | Headline |", "| --- | --- | --- |"])
            for panel in self.panels:
                lines.append(
                    f"| {panel.name} | {panel.status.value.upper()} | {panel.headline} |"
                )
            lines.append("")

        for panel in self.panels:
            lines.append(f"## {panel.name}")
            lines.append(f"**Status:** {panel.status.value.upper()}")
            lines.append(panel.headline)
            if panel.details:
                lines.append("")
                lines.extend(f"- {detail}" for detail in panel.details)
            if panel.metadata:
                lines.append("")
                lines.append("Metadata:")
                for key, value in sorted(panel.metadata.items()):
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def _map_roi_status(status: RoiStatus) -> DashboardStatus:
    if status is RoiStatus.at_risk:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_event_bus_status(status: EventBusHealthStatus) -> DashboardStatus:
    if status is EventBusHealthStatus.fail:
        return DashboardStatus.fail
    if status is EventBusHealthStatus.warn:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_slo_status(status: SLOStatus) -> DashboardStatus:
    if status is SLOStatus.breached:
        return DashboardStatus.fail
    if status is SLOStatus.at_risk:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_backbone_status(status: BackboneStatus) -> DashboardStatus:
    if status is BackboneStatus.fail:
        return DashboardStatus.fail
    if status is BackboneStatus.warn:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_operational_readiness_status(
    status: OperationalReadinessStatus,
) -> DashboardStatus:
    if status is OperationalReadinessStatus.fail:
        return DashboardStatus.fail
    if status is OperationalReadinessStatus.warn:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_quality_status(status: QualityStatus) -> DashboardStatus:
    if status is QualityStatus.fail:
        return DashboardStatus.fail
    if status is QualityStatus.warn:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _normalise_risk_result(result: Any) -> Mapping[str, Any]:
    if hasattr(result, "as_dict"):
        payload = dict(result.as_dict())  # type: ignore[call-arg]
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {"value": float(result)}

    normalised: MutableMapping[str, Any] = {}
    if "value" in payload:
        try:
            normalised["value"] = float(payload["value"])
        except (TypeError, ValueError):
            pass
    if "confidence" in payload and payload["confidence"] is not None:
        try:
            normalised["confidence"] = float(payload["confidence"])
        except (TypeError, ValueError):
            pass
    if "sample_size" in payload and payload["sample_size"] is not None:
        try:
            normalised["sample_size"] = int(float(payload["sample_size"]))
        except (TypeError, ValueError):
            pass
    for key, value in payload.items():
        if key not in normalised:
            normalised[key] = value
    return normalised


def _summarise_components(
    components: Iterable[BackboneComponentSnapshot],
) -> tuple[str, ...]:
    summaries: list[str] = []
    for component in components:
        summaries.append(
            f"{component.name}: {component.status.value} — {component.summary}"
        )
    return tuple(summaries)


def build_observability_dashboard(
    *,
    roi_snapshot: RoiTelemetrySnapshot | None = None,
    risk_results: Mapping[str, Any] | None = None,
    risk_limits: Mapping[str, float] | None = None,
    event_bus_snapshot: EventBusHealthSnapshot | None = None,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    loop_slo_snapshot: OperationalSLOSnapshot | None = None,
    backbone_snapshot: DataBackboneReadinessSnapshot | None = None,
    operational_readiness_snapshot: OperationalReadinessSnapshot | None = None,
    quality_snapshot: QualityTelemetrySnapshot | None = None,
    additional_panels: Sequence[DashboardPanel] | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObservabilityDashboard:
    """Compose the high-impact observability dashboard."""

    panels: list[DashboardPanel] = list(additional_panels or [])

    if roi_snapshot is not None:
        roi_status = _map_roi_status(roi_snapshot.status)
        headline = (
            "Net ROI {roi:.2%} (annualised {annual:.2%}, target {target:.2%})"
        ).format(
            roi=roi_snapshot.roi,
            annual=roi_snapshot.annualised_roi,
            target=roi_snapshot.target_annual_roi,
        )
        details = (
            "Equity {equity} vs initial {initial}".format(
                equity=_format_currency(roi_snapshot.current_equity),
                initial=_format_currency(roi_snapshot.initial_capital),
            ),
            "PnL gross {gross} / net {net}".format(
                gross=_format_currency(roi_snapshot.gross_pnl),
                net=_format_currency(roi_snapshot.net_pnl),
            ),
            "Trades {trades} — notional {notional}".format(
                trades=roi_snapshot.executed_trades,
                notional=_format_currency(roi_snapshot.total_notional),
            ),
            "Cost drag infra {infra} / fees {fees}".format(
                infra=_format_currency(roi_snapshot.infrastructure_cost),
                fees=_format_currency(roi_snapshot.fees),
            ),
        )
        panels.append(
            DashboardPanel(
                name="PnL & ROI",
                status=roi_status,
                headline=headline,
                details=details,
                metadata={"roi": roi_snapshot.as_dict()},
            )
        )

    if risk_results:
        risk_status = DashboardStatus.ok
        lines: list[str] = []
        serialised: dict[str, Mapping[str, Any]] = {}
        worst_name: str | None = None
        worst_value = 0.0
        worst_ratio: float | None = None

        for name, result in sorted(risk_results.items()):
            payload = dict(_normalise_risk_result(result))
            value = float(payload.get("value", 0.0))
            confidence = payload.get("confidence")
            sample_size = payload.get("sample_size")
            line = f"{name}: {value:,.2f}"
            if isinstance(confidence, (int, float)):
                line += f" @ {float(confidence):.1%}"
            if isinstance(sample_size, (int, float)):
                line += f" (n={int(sample_size)})"

            limit = float(risk_limits.get(name)) if risk_limits and name in risk_limits else None
            ratio: float | None = None
            limit_status = "unknown"
            if limit is not None and limit > 0.0:
                ratio = value / limit
                line += f" limit {limit:,.2f}"
                if value > limit:
                    risk_status = DashboardStatus.fail
                    limit_status = "violation"
                elif value > limit * 0.8:
                    risk_status = _escalate(risk_status, DashboardStatus.warn)
                    limit_status = "warn"
                else:
                    limit_status = "ok"
                payload["limit"] = limit
                payload["limit_ratio"] = ratio
                payload["limit_status"] = limit_status
            elif limit is not None:
                payload["limit"] = limit
                payload["limit_status"] = "invalid"

            if value > worst_value:
                worst_value = value
                worst_name = name
                worst_ratio = ratio

            lines.append(line)
            serialised[name] = payload

        if worst_name is not None:
            if worst_ratio is not None:
                headline = (
                    f"{worst_name} {worst_value:,.2f} "
                    f"({worst_ratio:.0%} of limit)"
                )
            else:
                headline = f"Peak tail risk {worst_value:,.2f}"
        else:
            headline = "Risk metrics unavailable"

        panels.append(
            DashboardPanel(
                name="Risk & exposure",
                status=risk_status,
                headline=headline,
                details=tuple(lines),
                metadata=serialised,
            )
        )

    if event_bus_snapshot is not None or slo_snapshot is not None or loop_slo_snapshot is not None:
        latency_status = DashboardStatus.ok
        details: list[str] = []
        latency_metadata: dict[str, Any] = {}
        headline_parts: list[str] = []

        if event_bus_snapshot is not None:
            latency_status = _map_event_bus_status(event_bus_snapshot.status)
            details.append(
                "Event bus queue {size} (dropped {dropped}, errors {errors})".format(
                    size=event_bus_snapshot.queue_size,
                    dropped=event_bus_snapshot.dropped_events,
                    errors=event_bus_snapshot.handler_errors,
                )
            )
            if event_bus_snapshot.queue_capacity is not None:
                details.append(
                    f"Queue capacity {event_bus_snapshot.queue_capacity}"
                )
            if event_bus_snapshot.last_event_at is not None:
                details.append(
                    f"Last event: {event_bus_snapshot.last_event_at.isoformat()}"
                )
            if event_bus_snapshot.last_error_at is not None:
                details.append(
                    f"Last error: {event_bus_snapshot.last_error_at.isoformat()}"
                )
            if event_bus_snapshot.issues:
                details.extend(f"Issue: {issue}" for issue in event_bus_snapshot.issues)
            latency_metadata["event_bus"] = event_bus_snapshot.as_dict()
            headline_parts.append(f"Event bus {event_bus_snapshot.status.value}")

        if slo_snapshot is not None:
            slo_status = _map_slo_status(slo_snapshot.status)
            latency_status = _escalate(latency_status, slo_status)
            counts = Counter(slo.status for slo in slo_snapshot.slos)
            details.append(
                "SLOs pass {ok} / warn {warn} / fail {fail}".format(
                    ok=counts.get(SLOStatus.met, 0),
                    warn=counts.get(SLOStatus.at_risk, 0),
                    fail=counts.get(SLOStatus.breached, 0),
                )
            )
            latency_metadata["slos"] = slo_snapshot.as_dict()
            headline_parts.append(f"Ingest SLOs {slo_snapshot.status.value}")

        if loop_slo_snapshot is not None:
            loop_status = _map_slo_status(loop_slo_snapshot.status)
            latency_status = _escalate(latency_status, loop_status)
            counts = Counter(slo.status for slo in loop_slo_snapshot.slos)
            details.append(
                "Loop SLOs pass {ok} / warn {warn} / fail {fail}".format(
                    ok=counts.get(SLOStatus.met, 0),
                    warn=counts.get(SLOStatus.at_risk, 0),
                    fail=counts.get(SLOStatus.breached, 0),
                )
            )
            latency_metadata["loop_slos"] = loop_slo_snapshot.as_dict()
            headline_parts.append(f"Loop SLOs {loop_slo_snapshot.status.value}")

        headline = "; ".join(headline_parts) if headline_parts else "Latency overview"

        panels.append(
            DashboardPanel(
                name="Latency & throughput",
                status=latency_status,
                headline=headline,
                details=tuple(details),
                metadata=latency_metadata,
            )
        )

    if backbone_snapshot is not None:
        backbone_status = _map_backbone_status(backbone_snapshot.status)
        component_summary = _summarise_components(backbone_snapshot.components)
        counts = Counter(component.status for component in backbone_snapshot.components)
        details = [
            "Components OK {ok} / warn {warn} / fail {fail}".format(
                ok=counts.get(BackboneStatus.ok, 0),
                warn=counts.get(BackboneStatus.warn, 0),
                fail=counts.get(BackboneStatus.fail, 0),
            )
        ]
        if component_summary:
            details.append("")
            details.extend(component_summary)

        metadata_payload = backbone_snapshot.as_dict()
        panels.append(
            DashboardPanel(
                name="System health",
                status=backbone_status,
                headline=f"Backbone status {backbone_snapshot.status.value}",
                details=tuple(details),
                metadata={"backbone": metadata_payload},
            )
        )

    if operational_readiness_snapshot is not None:
        readiness_status = _map_operational_readiness_status(
            operational_readiness_snapshot.status
        )
        component_counts = Counter(
            component.status for component in operational_readiness_snapshot.components
        )
        details: list[str] = [
            "Components OK {ok} / warn {warn} / fail {fail}".format(
                ok=component_counts.get(OperationalReadinessStatus.ok, 0),
                warn=component_counts.get(OperationalReadinessStatus.warn, 0),
                fail=component_counts.get(OperationalReadinessStatus.fail, 0),
            )
        ]
        degraded_components = [
            f"{component.name}: {component.summary}"
            for component in operational_readiness_snapshot.components
            if component.status is not OperationalReadinessStatus.ok
        ]
        if degraded_components:
            details.append("")
            details.extend(degraded_components)

        panels.append(
            DashboardPanel(
                name="Operational readiness",
                status=readiness_status,
                headline=(
                    f"Readiness {operational_readiness_snapshot.status.value}"
                ),
                details=tuple(details),
                metadata={
                    "operational_readiness": operational_readiness_snapshot.as_dict()
                },
            )
        )

    if quality_snapshot is not None:
        quality_status = _map_quality_status(quality_snapshot.status)
        details = list(quality_snapshot.notes)
        if quality_snapshot.remediation_items:
            details.append("")
            details.extend(
                f"Remediation: {item}" for item in quality_snapshot.remediation_items
            )

        if quality_snapshot.coverage_percent is None:
            headline = "Coverage telemetry unavailable"
        else:
            headline = (
                f"Coverage {quality_snapshot.coverage_percent:.2f}%"
                f" (target {quality_snapshot.coverage_target:.2f}%)"
            )

        panels.append(
            DashboardPanel(
                name="Quality & coverage",
                status=quality_status,
                headline=headline,
                details=tuple(details),
                metadata={"quality": quality_snapshot.as_dict()},
            )
        )

    overall_status = DashboardStatus.ok
    for panel in panels:
        overall_status = _escalate(overall_status, panel.status)

    dashboard_metadata = dict(metadata or {})
    status_counts = Counter(panel.status for panel in panels)
    dashboard_metadata.setdefault(
        "panel_status_counts",
        {
            DashboardStatus.ok.value: status_counts.get(DashboardStatus.ok, 0),
            DashboardStatus.warn.value: status_counts.get(DashboardStatus.warn, 0),
            DashboardStatus.fail.value: status_counts.get(DashboardStatus.fail, 0),
        },
    )
    dashboard_metadata.setdefault(
        "panel_statuses",
        {panel.name: panel.status.value for panel in panels},
    )
    if generated_at is None:
        generated_at = datetime.now(tz=UTC)

    return ObservabilityDashboard(
        generated_at=generated_at,
        status=overall_status,
        panels=tuple(panels),
        metadata=dashboard_metadata,
    )
