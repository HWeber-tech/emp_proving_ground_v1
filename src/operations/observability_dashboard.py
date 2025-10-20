from __future__ import annotations

"""Textual observability dashboard aligned with the high-impact roadmap."""

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import PolicyLedgerStage
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)
from src.operations.event_bus_health import (
    EventBusHealthSnapshot,
    EventBusHealthStatus,
)
from src.operations.evolution_kpis import EvolutionKpiSnapshot, EvolutionKpiStatus
from src.operations.gate_dashboard import (
    GateDashboard,
    GateDashboardStatus,
    build_gate_dashboard,
    build_gate_dashboard_from_mapping,
)
from src.operations.operator_leverage import (
    OperatorLeverageSnapshot,
    OperatorLeverageStatus,
)
from src.operations.operational_readiness import (
    OperationalReadinessSnapshot,
    OperationalReadinessStatus,
)
from src.operational import metrics as operational_metrics
from src.operations.quality_telemetry import (
    QualityStatus,
    QualityTelemetrySnapshot,
)
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot
from src.operations.slo import OperationalSLOSnapshot, SLOStatus, ServiceSLO
from src.operations.sensory_drift import DriftSeverity, SensoryDriftSnapshot
from src.understanding.diagnostics import (
    UnderstandingGraphStatus,
    UnderstandingLoopSnapshot,
)
from src.understanding.metrics import export_understanding_throttle_metrics

if TYPE_CHECKING:
    from src.governance.policy_graduation import PolicyGraduationAssessment
    from src.orchestration.alpha_trade_loop import (
        AlphaTradeLoopResult,
        ComplianceEvent,
    )
    from src.thinking.adaptation.policy_reflection import PolicyReflectionArtifacts

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback for older runtimes
    UTC = timezone.utc


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


_LEDGER_STAGE_ORDER: Mapping[PolicyLedgerStage, int] = {
    PolicyLedgerStage.EXPERIMENT: 0,
    PolicyLedgerStage.PAPER: 1,
    PolicyLedgerStage.PILOT: 2,
    PolicyLedgerStage.LIMITED_LIVE: 3,
}

_GATE_STATUS_TO_DASHBOARD: Mapping[GateDashboardStatus, DashboardStatus] = {
    GateDashboardStatus.OK: DashboardStatus.ok,
    GateDashboardStatus.WARN: DashboardStatus.warn,
    GateDashboardStatus.FAIL: DashboardStatus.fail,
}


def _escalate(
    current: DashboardStatus, candidate: DashboardStatus
) -> DashboardStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_percent(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1.0 + 1e-9:
        return value
    return value * 100.0


def _format_ms(value: Any) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return "n/a"
    if numeric >= 1000.0:
        return f"{numeric / 1000.0:.2f}s"
    if numeric >= 100.0:
        return f"{numeric:.0f}ms"
    return f"{numeric:.1f}ms"


def _format_rate(value: Any, unit: str) -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return f"n/a {unit}"
    return f"{numeric:.2f} {unit}"


def _format_memory(value: Any, *, unit: str = "MiB") -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return f"n/a {unit}"
    return f"{numeric:.1f} {unit}"


def _map_monitoring_status(value: Any) -> DashboardStatus:
    if isinstance(value, DashboardStatus):
        return value
    if isinstance(value, DriftSeverity):
        return _map_drift_status(value)
    if value is None:
        return DashboardStatus.ok
    text = str(value).strip().lower()
    if not text:
        return DashboardStatus.ok
    fail_tokens = {
        "fail",
        "critical",
        "alert",
        "severe",
        "fatal",
        "breach",
        "error",
    }
    warn_tokens = {
        "warn",
        "warning",
        "at_risk",
        "caution",
        "degraded",
        "anomaly",
    }
    if text in fail_tokens:
        return DashboardStatus.fail
    if text in warn_tokens:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_drift_status(value: DriftSeverity | str | None) -> DashboardStatus:
    if isinstance(value, DriftSeverity):
        if value is DriftSeverity.alert:
            return DashboardStatus.fail
        if value is DriftSeverity.warn:
            return DashboardStatus.warn
        return DashboardStatus.ok
    return _map_monitoring_status(value)


def _normalise_drift_payload(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, SensoryDriftSnapshot):
        return value.as_dict()
    if isinstance(value, Mapping):
        payload = dict(value)
        dimensions = payload.get("dimensions")
        if isinstance(dimensions, Mapping):
            payload["dimensions"] = {
                str(name): dict(dimension)
                for name, dimension in dimensions.items()
                if isinstance(dimension, Mapping)
            }
        return payload
    return None


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


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Telemetry slice capturing runtime monitoring metrics."""

    generated_at: datetime
    latency_ms: Mapping[str, Any] = field(default_factory=dict)
    throughput: Mapping[str, Any] = field(default_factory=dict)
    pnl: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    tail_spikes: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    drift_summary: Mapping[str, Any] | SensoryDriftSnapshot | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "latency_ms": dict(self.latency_ms),
            "throughput": dict(self.throughput),
            "pnl": dict(self.pnl),
            "memory": dict(self.memory),
            "metadata": dict(self.metadata),
        }
        tail_alerts = [dict(entry) for entry in self.tail_spikes if isinstance(entry, Mapping)]
        if tail_alerts:
            payload["tail_spikes"] = tail_alerts
        else:
            payload["tail_spikes"] = []

        if self.drift_summary is not None:
            if isinstance(self.drift_summary, SensoryDriftSnapshot):
                payload["drift_summary"] = self.drift_summary.as_dict()
            elif isinstance(self.drift_summary, Mapping):
                payload["drift_summary"] = dict(self.drift_summary)
        return payload


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


def _map_compliance_status(value: str | None) -> DashboardStatus:
    if value in {"critical", "fail", "alert"}:
        return DashboardStatus.fail
    if value in {"warn", "warning"}:
        return DashboardStatus.warn
    return DashboardStatus.ok


def _map_evolution_status(value: EvolutionKpiStatus | str | None) -> DashboardStatus:
    if isinstance(value, EvolutionKpiStatus):
        resolved = value.value
    elif value is None:
        return DashboardStatus.warn
    else:
        resolved = str(value).strip().lower()

    if resolved == EvolutionKpiStatus.fail.value:
        return DashboardStatus.fail
    if resolved in {EvolutionKpiStatus.warn.value, "warning"}:
        return DashboardStatus.warn
    if resolved == EvolutionKpiStatus.ok.value:
        return DashboardStatus.ok
    return DashboardStatus.warn


def _map_operator_leverage_status(value: OperatorLeverageStatus | str | None) -> DashboardStatus:
    if isinstance(value, OperatorLeverageStatus):
        resolved = value.value
    elif value is None:
        return DashboardStatus.warn
    else:
        resolved = str(value).strip().lower()

    if resolved == OperatorLeverageStatus.fail.value:
        return DashboardStatus.fail
    if resolved == OperatorLeverageStatus.warn.value:
        return DashboardStatus.warn
    if resolved == OperatorLeverageStatus.ok.value:
        return DashboardStatus.ok
    return DashboardStatus.warn


def _event_attr(event: Any, name: str) -> Any:
    if hasattr(event, name):
        candidate = getattr(event, name)
        return getattr(candidate, "value", candidate)
    if isinstance(event, Mapping):
        candidate = event.get(name)
        if isinstance(candidate, Mapping) and "value" in candidate:
            return candidate["value"]
        return candidate
    return None


def _event_as_dict(event: Any) -> Mapping[str, Any]:
    if hasattr(event, "as_dict"):
        try:
            payload = dict(event.as_dict())  # type: ignore[call-arg]
        except TypeError:
            payload = {}
    elif isinstance(event, Mapping):
        payload = dict(event)
    else:
        payload = {}
    if "event_type" not in payload:
        payload["event_type"] = _event_attr(event, "event_type")
    if "severity" not in payload:
        payload["severity"] = _event_attr(event, "severity")
    if "summary" not in payload:
        payload["summary"] = _event_attr(event, "summary") or ""
    if "policy_id" not in payload:
        payload["policy_id"] = _event_attr(event, "policy_id")
    occurred_at = payload.get("occurred_at")
    if occurred_at is None:
        occurred_at = _event_attr(event, "occurred_at")
    payload["occurred_at"] = occurred_at
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        payload["metadata"] = {}
    return payload


def _extract_diary_coverage(
    loop_results: Sequence[Any] | None,
) -> Mapping[str, Any] | None:
    if not loop_results:
        return None

    candidates: list[Mapping[str, Any]] = []
    for result in loop_results:
        candidate: Mapping[str, Any] | None = None
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, Mapping):
            candidate = metadata.get("diary_coverage")  # type: ignore[assignment]
        if candidate is None and isinstance(result, Mapping):
            metadata_mapping = result.get("metadata") if "metadata" in result else None
            if isinstance(metadata_mapping, Mapping):
                candidate = metadata_mapping.get("diary_coverage")  # type: ignore[assignment]
            elif isinstance(result.get("diary_coverage"), Mapping):
                candidate = result.get("diary_coverage")  # type: ignore[assignment]
        if candidate is None:
            trade_metadata = getattr(result, "trade_metadata", None)
            if isinstance(trade_metadata, Mapping):
                candidate = trade_metadata.get("diary_coverage")  # type: ignore[assignment]
        if isinstance(candidate, Mapping) and candidate:
            candidates.append(candidate)

    if not candidates:
        return None

    return dict(candidates[-1])


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _derive_compliance_slo(
    events: Sequence[Any],
    *,
    generated_at: datetime | None = None,
) -> OperationalSLOSnapshot:
    generated = generated_at or datetime.now(tz=UTC)
    breach_count = 0
    warn_count = 0
    governance_actions = 0
    promotions = 0

    for event in events:
        event_type = str(_event_attr(event, "event_type") or "").lower()
        if event_type == "risk_breach":
            breach_count += 1
        elif event_type == "risk_warning":
            warn_count += 1
        elif event_type == "governance_promotion":
            promotions += 1
        elif event_type == "governance_action":
            governance_actions += 1

    breach_status = SLOStatus.met
    breach_message = "No risk breaches observed"
    if breach_count > 0:
        breach_status = SLOStatus.breached
        breach_message = f"{breach_count} risk breach(es) intercepted"
    elif warn_count > 0:
        breach_status = SLOStatus.at_risk
        breach_message = f"{warn_count} risk warning(s) under observation"

    governance_status = SLOStatus.met
    if governance_actions > 0:
        governance_status = SLOStatus.at_risk
        governance_message = (
            f"{governance_actions} governance intervention(s) require review"
        )
    else:
        governance_message = "No governance interventions recorded"

    slos = (
        ServiceSLO(
            name="policy_breach_prevention",
            status=breach_status,
            message=breach_message,
            target={"breaches": 0},
            observed={
                "breaches": breach_count,
                "warnings": warn_count,
            },
        ),
        ServiceSLO(
            name="governance_interventions",
            status=governance_status,
            message=governance_message,
            target={"interventions": 0},
            observed={
                "interventions": governance_actions,
                "promotions": promotions,
            },
        ),
    )

    if any(record.status is SLOStatus.breached for record in slos):
        aggregate_status = SLOStatus.breached
    elif any(record.status is SLOStatus.at_risk for record in slos):
        aggregate_status = SLOStatus.at_risk
    else:
        aggregate_status = SLOStatus.met

    return OperationalSLOSnapshot(
        service="risk_compliance",
        generated_at=generated,
        status=aggregate_status,
        slos=slos,
        metadata={
            "breaches": breach_count,
            "warnings": warn_count,
            "governance_interventions": governance_actions,
            "promotions": promotions,
        },
    )


def _build_compliance_panel(
    events: Sequence[Any],
    *,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    generated_at: datetime | None = None,
) -> DashboardPanel:
    sorted_events = sorted(
        events,
        key=lambda event: _coerce_timestamp(_event_attr(event, "occurred_at"))
        or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )

    has_events = bool(events)
    risk_breaches = 0
    risk_warnings = 0
    governance_actions = 0
    governance_promotions = 0
    details: list[str] = []
    metadata: MutableMapping[str, Any] = {
        "events": [],
        "counts": {},
    }
    panel_status = DashboardStatus.ok

    for event in sorted_events:
        payload = _event_as_dict(event)
        metadata["events"].append(payload)

        event_type = str(payload.get("event_type") or "").lower()
        severity = str(payload.get("severity") or "").lower()
        policy_id = payload.get("policy_id") or "policy"
        occurred_at = _coerce_timestamp(payload.get("occurred_at"))
        timestamp = occurred_at.isoformat() if occurred_at else "unknown"
        summary = payload.get("summary") or "Compliance event recorded"

        panel_status = _escalate(panel_status, _map_compliance_status(severity))

        if event_type == "risk_breach":
            risk_breaches += 1
        elif event_type == "risk_warning":
            risk_warnings += 1
        elif event_type == "governance_promotion":
            governance_promotions += 1
        elif event_type == "governance_action":
            governance_actions += 1

        details.append(f"{timestamp} — {policy_id}: {summary}")

    metadata["counts"] = {
        "risk_breach": risk_breaches,
        "risk_warning": risk_warnings,
        "governance_action": governance_actions,
        "governance_promotion": governance_promotions,
    }

    slo = slo_snapshot or _derive_compliance_slo(sorted_events, generated_at=generated_at)
    metadata["compliance_slo"] = slo.as_dict()
    panel_status = _escalate(panel_status, _map_slo_status(slo.status))

    if slo.slos:
        details.append("")
        details.append("SLO overview:")
        for record in slo.slos:
            details.append(
                f"- {record.name}: {record.status.value} — {record.message}"
            )

    if not has_events:
        panel_status = _escalate(panel_status, DashboardStatus.warn)
        details.insert(
            0,
            "No compliance telemetry supplied; provide loop results or explicit compliance events.",
        )

    headline_parts: list[str] = []
    if risk_breaches:
        headline_parts.append(f"Risk breaches {risk_breaches}")
    if risk_warnings:
        headline_parts.append(f"Risk warnings {risk_warnings}")
    if governance_actions:
        headline_parts.append(f"Governance actions {governance_actions}")
    if governance_promotions:
        headline_parts.append(f"Promotions {governance_promotions}")

    if not headline_parts:
        headline = "Compliance telemetry baseline"
    else:
        headline = " | ".join(headline_parts)

    operational_metrics.set_compliance_policy_breaches(float(risk_breaches))
    operational_metrics.set_compliance_risk_warnings(float(risk_warnings))
    operational_metrics.set_governance_actions_total(float(governance_actions))
    operational_metrics.set_governance_promotions_total(float(governance_promotions))

    return DashboardPanel(
        name="Compliance & governance",
        status=panel_status,
        headline=headline,
        details=tuple(details),
        metadata=dict(metadata),
    )


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


def _map_understanding_status(status: UnderstandingGraphStatus) -> DashboardStatus:
    if status is UnderstandingGraphStatus.fail:
        return DashboardStatus.fail
    if status is UnderstandingGraphStatus.warn:
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


def _format_hours(value: object) -> str:
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{hours:.1f}h"


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


def _format_percentage(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _first_mapping(entry: object) -> Mapping[str, Any] | None:
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        for element in entry:
            if isinstance(element, Mapping):
                return element
    return None


def _build_evolution_panel(
    snapshot: EvolutionKpiSnapshot | Mapping[str, Any] | None,
) -> DashboardPanel:
    if snapshot is None:
        return DashboardPanel(
            name="Evolution KPIs",
            status=DashboardStatus.warn,
            headline="Evolution KPI telemetry unavailable",
            details=(
                "Supply evolution KPI snapshots via operations.evolution_kpis "
                "to populate the panel with live metrics.",
            ),
            metadata={"evolution_kpis": {"status": "missing"}},
        )

    if isinstance(snapshot, EvolutionKpiSnapshot):
        snapshot_dict = snapshot.as_dict()
    else:
        snapshot_dict = dict(snapshot)

    panel_status = _map_evolution_status(snapshot_dict.get("status"))

    time_data = snapshot_dict.get("time_to_candidate") or {}
    promotion_data = snapshot_dict.get("promotion") or {}
    budget_data = snapshot_dict.get("budget") or {}
    rollback_data = snapshot_dict.get("rollback") or {}

    details: list[str] = []
    headline_parts: list[str] = []

    if time_data:
        average = _format_hours(time_data.get("avg_hours"))
        p90 = _format_hours(
            time_data.get("p90_hours") or time_data.get("p90")
        )
        threshold = _format_hours(time_data.get("threshold_hours"))
        sla_met = time_data.get("sla_met")
        breaches = time_data.get("breaches")
        breach_count = len(breaches) if isinstance(breaches, Sequence) else 0
        headline_parts.append(f"TTC avg {average}")
        details.append(
            "Time-to-candidate: avg {avg}, p90 {p90}, threshold {threshold}, SLA {sla}".format(
                avg=average,
                p90=p90,
                threshold=threshold,
                sla="PASS" if sla_met else "FAIL",
            )
        )
        counts_line = "Ideas evaluated {count}".format(
            count=int(time_data.get("count", 0) or 0)
        )
        if breach_count:
            counts_line += f" — breaches {breach_count}"
        details.append(counts_line)

    if promotion_data:
        rate_value = promotion_data.get("promotion_rate")
        rate_text = _format_percentage(rate_value)
        promotions = int(promotion_data.get("promotions", 0) or 0)
        transitions = int(promotion_data.get("transitions", 0) or 0)
        demotions = int(promotion_data.get("demotions", 0) or 0)
        headline_parts.append(f"Promotion rate {rate_text}")
        details.append(
            "Promotion posture: rate {rate} "
            "({promotions}/{transitions} promotions, demotions {demotions})".format(
                rate=rate_text,
                promotions=promotions,
                transitions=transitions,
                demotions=demotions,
            )
        )

    if budget_data:
        avg_usage = _format_percentage(budget_data.get("average_usage_ratio"))
        max_usage = _format_percentage(budget_data.get("max_usage_ratio"))
        blocked = int(budget_data.get("blocked_attempts", 0) or 0)
        forced = int(budget_data.get("forced_decisions", 0) or 0)
        headline_parts.append(f"Budget max {max_usage}")
        details.append(
            "Exploration budget: avg usage {avg}, max {max}, blocked {blocked}, forced {forced}".format(
                avg=avg_usage,
                max=max_usage,
                blocked=blocked,
                forced=forced,
            )
        )

    if rollback_data:
        samples = int(rollback_data.get("samples", 0) or 0)
        if samples:
            median = _format_hours(rollback_data.get("median_hours"))
            max_latency = _format_hours(rollback_data.get("max_hours"))
            headline_parts.append(f"Rollback median {median}")
            details.append(
                f"Rollback latency: median {median}, max {max_latency}, samples {samples}"
            )
        else:
            details.append("Rollback latency: no samples recorded")

    if not details:
        details.append(
            "No evolution KPI metrics supplied; export operations.evolution_kpis.EvolutionKpiSnapshot."
        )
        panel_status = _map_evolution_status(None)

    headline = " | ".join(headline_parts) if headline_parts else "Evolution KPI snapshot"

    return DashboardPanel(
        name="Evolution KPIs",
        status=panel_status,
        headline=headline,
        details=tuple(details),
        metadata={"evolution_kpis": snapshot_dict},
    )


def _build_operator_leverage_panel(
    snapshot: OperatorLeverageSnapshot | Mapping[str, Any] | None,
) -> DashboardPanel:
    if snapshot is None:
        return DashboardPanel(
            name="Operator leverage",
            status=DashboardStatus.warn,
            headline="Operator leverage telemetry unavailable",
            details=(
                "Provide operator leverage snapshots via operations.operator_leverage "
                "to track experiments per week and quality posture.",
            ),
            metadata={"operator_leverage": {"status": "missing"}},
        )

    if isinstance(snapshot, OperatorLeverageSnapshot):
        snapshot_dict = snapshot.as_dict()
    else:
        snapshot_dict = dict(snapshot)

    panel_status = _map_operator_leverage_status(snapshot_dict.get("status"))
    operators = [
        entry for entry in snapshot_dict.get("operators", []) if isinstance(entry, Mapping)
    ]
    operator_count = snapshot_dict.get("operator_count")
    if not isinstance(operator_count, int):
        operator_count = len(operators)

    avg_per_week = _coerce_float(snapshot_dict.get("experiments_per_week"))
    total_per_week = _coerce_float(snapshot_dict.get("experiments_per_week_total"))
    if avg_per_week is None and total_per_week is not None and operator_count:
        avg_per_week = total_per_week / max(operator_count, 1)

    quality_rate = _coerce_float(snapshot_dict.get("quality_pass_rate"))

    headline_parts: list[str] = []
    if operator_count:
        headline_parts.append(
            f"{operator_count} operator{'s' if operator_count != 1 else ''}"
        )
    if avg_per_week is not None:
        headline_parts.append(f"{avg_per_week:.2f}/wk avg")
    if total_per_week is not None and (avg_per_week is None or operator_count > 1):
        headline_parts.append(f"{total_per_week:.2f}/wk total")
    if quality_rate is not None:
        headline_parts.append(f"quality {quality_rate:.0%}")
    else:
        headline_parts.append("quality n/a")

    details: list[str] = []
    if operators:
        sorted_ops = sorted(
            operators,
            key=lambda entry: (
                _coerce_float(entry.get("experiments_per_week")) or 0.0,
                float(entry.get("experiments", 0) or 0.0),
            ),
            reverse=True,
        )
        top_lines: list[str] = []
        for entry in sorted_ops[:3]:
            name = str(entry.get("operator", "unknown"))
            per_week_value = _coerce_float(entry.get("experiments_per_week")) or 0.0
            quality_value = _coerce_float(entry.get("quality_rate"))
            quality_text = f"{quality_value:.0%}" if quality_value is not None else "n/a"
            top_lines.append(f"{name}: {per_week_value:.2f}/wk (quality {quality_text})")
        details.append("Top operators: " + "; ".join(top_lines))
    else:
        details.append("No experiments recorded in the lookback window.")

    meta = snapshot_dict.get("metadata")
    if isinstance(meta, Mapping):
        low_velocity_fail = [str(op) for op in meta.get("low_velocity_fail", ()) if op]
        if low_velocity_fail:
            details.append("Velocity FAIL: " + ", ".join(low_velocity_fail))
        low_velocity_warn = [str(op) for op in meta.get("low_velocity_warn", ()) if op]
        low_velocity_warn = [op for op in low_velocity_warn if op not in low_velocity_fail]
        if low_velocity_warn:
            details.append("Velocity WARN: " + ", ".join(low_velocity_warn))
        quality_fail = [str(op) for op in meta.get("quality_fail", ()) if op]
        if quality_fail:
            details.append("Quality FAIL: " + ", ".join(quality_fail))
        quality_warn = [str(op) for op in meta.get("quality_warn", ()) if op]
        quality_warn = [op for op in quality_warn if op not in quality_fail]
        if quality_warn:
            details.append("Quality WARN: " + ", ".join(quality_warn))
        missing_quality = [str(op) for op in meta.get("quality_missing", ()) if op]
        if missing_quality:
            details.append("Quality missing: " + ", ".join(missing_quality))
        failure_reasons = meta.get("top_failure_reasons")
        if isinstance(failure_reasons, Mapping) and failure_reasons:
            summary = ", ".join(
                f"{reason}: {count}"
                for reason, count in failure_reasons.items()
            )
            details.append("Top failure reasons: " + summary)

    headline = " | ".join(headline_parts) if headline_parts else "Operator leverage snapshot"

    return DashboardPanel(
        name="Operator leverage",
        status=panel_status,
        headline=headline,
        details=tuple(details),
        metadata={"operator_leverage": snapshot_dict},
    )


def _build_diary_panel(loop_results: Sequence[Any] | None) -> DashboardPanel:
    coverage_payload = _extract_diary_coverage(loop_results)
    if coverage_payload is None:
        return DashboardPanel(
            name="Decision diary",
            status=DashboardStatus.warn,
            headline="Diary coverage telemetry unavailable",
            details=(
                "No diary coverage metadata emitted; ensure AlphaTrade loop results include "
                "runner.describe_diary_coverage() snapshots.",
            ),
            metadata={"decision_diary": {"status": "missing"}},
        )

    info = dict(coverage_payload)

    coverage_value = _coerce_float(info.get("coverage"))
    coverage_percent_input = _coerce_float(info.get("coverage_percent"))
    coverage_inferred = info.get("coverage_inferred") is True
    coverage_reported = coverage_value is not None or coverage_percent_input is not None

    if coverage_value is not None and coverage_value > 1.0 + 1e-9:
        coverage_value = coverage_value / 100.0
        coverage_inferred = True

    if coverage_value is None and coverage_percent_input is not None:
        coverage_value = (
            coverage_percent_input / 100.0
            if coverage_percent_input > 1.0 + 1e-9
            else coverage_percent_input
        )
        coverage_inferred = True

    iterations_raw = info.get("iterations")
    iterations_value = _coerce_int(iterations_raw)
    iterations_provided = iterations_value is not None
    iterations = iterations_value or 0

    recorded_raw = info.get("recorded")
    recorded_value = _coerce_int(recorded_raw)
    recorded_provided = recorded_value is not None
    recorded = recorded_value or 0

    if (
        coverage_value is None
        and iterations_provided
        and recorded_provided
        and iterations > 0
    ):
        coverage_value = min(max(recorded / iterations, 0.0), 1.0)
        coverage_inferred = True

    if coverage_value is not None:
        info["coverage"] = coverage_value
    if coverage_inferred:
        info["coverage_inferred"] = True

    target_value = _coerce_float(info.get("target"))
    target_percent_input = _coerce_float(info.get("target_percent"))
    target_inferred = info.get("target_inferred") is True
    target_reported = target_value is not None or target_percent_input is not None

    if target_value is not None and target_value > 1.0 + 1e-9:
        target_value = target_value / 100.0

    if target_value is None and target_percent_input is not None:
        target_value = (
            target_percent_input / 100.0
            if target_percent_input > 1.0 + 1e-9
            else target_percent_input
        )

    if target_value is None:
        target_value = 0.95
        target_inferred = True

    info["target"] = target_value
    if target_inferred:
        info["target_inferred"] = True

    missing_raw = info.get("missing")
    missing_value = _coerce_int(missing_raw)
    if missing_value is not None:
        missing_value = max(missing_value, 0)
        info["missing"] = missing_value
    elif iterations_provided and recorded_provided:
        inferred_missing = max(iterations - recorded, 0)
        info["missing"] = inferred_missing
        missing_value = inferred_missing

    missing = missing_value or 0

    minimum_samples = _coerce_int(info.get("minimum_samples")) or 0
    gap_threshold = _coerce_float(info.get("gap_threshold_seconds"))
    gap_breach = bool(info.get("gap_breach"))
    last_recorded_at = info.get("last_recorded_at")

    coverage_percent = _normalise_percent(coverage_value)
    target_percent = _normalise_percent(target_value)

    if coverage_percent is not None:
        info.setdefault("coverage_percent", coverage_percent)
    if target_percent is not None:
        info.setdefault("target_percent", target_percent)

    if coverage_percent is not None and target_percent is not None:
        headline = f"Coverage {coverage_percent:.2f}% (target {target_percent:.2f}%)"
    elif coverage_percent is not None:
        headline = f"Coverage {coverage_percent:.2f}%"
    elif target_percent is not None:
        headline = f"Coverage target {target_percent:.2f}%"
    else:
        headline = "Coverage telemetry unavailable"

    status = DashboardStatus.ok
    details: list[str] = []

    if iterations_provided and recorded_provided:
        detail_line = f"Recorded {recorded} of {iterations} iterations"
        if missing:
            detail_line += f"; missing {missing}"
        details.append(detail_line)
    elif iterations_provided:
        if iterations:
            details.append(f"Recorded telemetry missing for {iterations} iterations.")
        else:
            details.append("No loop iterations recorded yet.")
        status = _escalate(status, DashboardStatus.warn)
    elif recorded_provided:
        details.append(f"Recorded {recorded} diary entries; iteration telemetry missing.")
        status = _escalate(status, DashboardStatus.warn)

    insufficient_samples = False
    if minimum_samples and iterations_provided and iterations < minimum_samples:
        insufficient_samples = True
        details.append(
            f"Minimum sample {minimum_samples} not satisfied (observed {iterations})."
        )
        status = _escalate(status, DashboardStatus.warn)

    coverage_below_target = False
    if (
        coverage_value is not None
        and target_value is not None
        and coverage_value + 1e-9 < target_value
    ):
        coverage_below_target = True
        if coverage_percent is not None and target_percent is not None:
            shortfall = target_percent - coverage_percent
            details.append(f"Coverage shortfall {shortfall:.2f}pp vs target.")
        else:
            details.append("Coverage below target.")
        status = _escalate(status, DashboardStatus.fail)

    if missing and missing > 0 and not coverage_below_target:
        details.append(f"Missing {missing} diary entries across recent iterations.")

    if gap_breach:
        message = "Diary gap alert active"
        if gap_threshold is not None:
            message += f" (> {gap_threshold:.0f}s)"
        if last_recorded_at:
            message += f"; last recorded {last_recorded_at}"
        details.append(message)
        status = _escalate(status, DashboardStatus.fail)
    else:
        if last_recorded_at:
            details.append(f"Last recorded at {last_recorded_at}")
        if gap_threshold is not None:
            details.append(f"Gap threshold {gap_threshold:.0f}s monitored")

    missing_telemetry_fields: list[str] = []
    if not coverage_reported:
        missing_telemetry_fields.append("coverage")
    if not iterations_provided:
        missing_telemetry_fields.append("iterations")
    if not recorded_provided:
        missing_telemetry_fields.append("recorded")
    if not target_reported:
        missing_telemetry_fields.append("target")

    missing_field_names: tuple[str, ...] = tuple(sorted(missing_telemetry_fields))
    missing_telemetry = bool(missing_field_names)
    if missing_telemetry:
        fields = ", ".join(missing_field_names)
        details.append(f"Missing diary telemetry: {fields}.")
        status = _escalate(status, DashboardStatus.warn)

    info["missing_telemetry_fields"] = missing_field_names

    if not details:
        details.append("No diary coverage telemetry available.")

    metadata_payload = {
        "decision_diary": {
            "coverage": info,
            "alerts": {
                "coverage_below_target": coverage_below_target,
                "gap_breach": gap_breach,
                "insufficient_samples": insufficient_samples,
                "missing_telemetry": missing_telemetry,
            },
        }
    }

    if coverage_percent is None or target_percent is None:
        status = _escalate(status, DashboardStatus.warn)

    return DashboardPanel(
        name="Decision diary",
        status=status,
        headline=headline,
        details=tuple(details),
        metadata=metadata_payload,
    )


def _build_policy_reflection_panel(
    artifacts: "PolicyReflectionArtifacts",
) -> DashboardPanel:
    digest = dict(artifacts.digest)
    payload = dict(artifacts.payload)
    insights = tuple(
        str(item)
        for item in payload.get("insights", ())
        if isinstance(item, str) and item.strip()
    )
    total = int(digest.get("total_decisions", 0) or 0)
    status = DashboardStatus.ok if total > 0 else DashboardStatus.warn
    headline = (
        f"Policy reflections analysed {total} decision{'s' if total != 1 else ''}"
        if total
        else "Policy reflections awaiting decisions"
    )

    details: list[str] = []
    top_tactic = _first_mapping(digest.get("tactics"))
    if top_tactic:
        tactic_id = str(top_tactic.get("tactic_id", "unknown"))
        share = _format_percentage(top_tactic.get("share", 0.0))
        avg_score = float(top_tactic.get("avg_score", 0.0))
        last_seen = top_tactic.get("last_seen")
        last_seen_text = str(last_seen) if last_seen else "unknown"
        details.append(
            "Top tactic {tactic} ({share}, avg score {score:.3f}, last {last})".format(
                tactic=tactic_id,
                share=share,
                score=avg_score,
                last=last_seen_text,
            )
        )

    top_experiment = _first_mapping(digest.get("experiments"))
    if top_experiment:
        experiment_id = str(top_experiment.get("experiment_id", "unknown"))
        count = int(top_experiment.get("count", 0) or 0)
        share = _format_percentage(top_experiment.get("share", 0.0))
        gating_bits: list[str] = []
        regimes = top_experiment.get("regimes")
        if isinstance(regimes, Sequence) and regimes and not isinstance(regimes, (str, bytes)):
            gating_bits.append(
                "regimes " + ", ".join(str(regime) for regime in regimes if str(regime).strip())
            )
        min_conf = top_experiment.get("min_confidence")
        if isinstance(min_conf, (int, float)) and float(min_conf) > 0.0:
            gating_bits.append(f"confidence >= {float(min_conf):.2f}")
        rationale = top_experiment.get("rationale")
        rationale_text = str(rationale).strip() if isinstance(rationale, str) else ""
        descriptor = "; ".join(gating_bits)
        details.append(
            "Top experiment {experiment} applied {count}x ({share}{gating}){rationale}".format(
                experiment=experiment_id,
                count=count,
                share=share,
                gating=f"; {descriptor}" if descriptor else "",
                rationale=f" - {rationale_text}" if rationale_text else "",
            )
        )

    top_tag = _first_mapping(digest.get("tags"))
    if top_tag:
        tag_name = str(top_tag.get("tag", "unknown"))
        share = _format_percentage(top_tag.get("share", 0.0))
        details.append(f"Dominant tag {tag_name} at {share}")

    top_objective = _first_mapping(digest.get("objectives"))
    if top_objective:
        objective_name = str(top_objective.get("objective", "unknown"))
        share = _format_percentage(top_objective.get("share", 0.0))
        details.append(f"Leading objective {objective_name} at {share}")

    headlines = digest.get("recent_headlines")
    if isinstance(headlines, Sequence) and headlines and not isinstance(headlines, (str, bytes)):
        if details:
            details.append("")
        details.extend(
            f"Headline: {str(headline)}"
            for headline in headlines[-3:]
            if str(headline).strip()
        )

    if insights:
        if details:
            details.append("")
        details.extend(f"Insight: {insight}" for insight in insights[:5])

    if not details:
        details.append("No reflection insights available; capture decision telemetry.")

    metadata_payload: dict[str, Any] = {
        "metadata": dict(payload.get("metadata", {})),
        "digest": digest,
        "insights": list(insights),
        "markdown": artifacts.markdown,
    }

    return DashboardPanel(
        name="Policy reflections",
        status=status,
        headline=headline,
        details=tuple(details),
        metadata={"policy_reflection": metadata_payload},
    )


def _normalise_graduation_assessments(
    assessments: Sequence["PolicyGraduationAssessment"]
    | Mapping[str, "PolicyGraduationAssessment"],
) -> list["PolicyGraduationAssessment"]:
    if isinstance(assessments, Mapping):
        candidates = assessments.values()
    else:
        candidates = assessments
    candidate_list = list(candidates)
    try:
        from src.governance.policy_graduation import (  # local import to avoid cycles
            PolicyGraduationAssessment as _PolicyGraduationAssessment,
        )
    except Exception:
        _PolicyGraduationAssessment = None  # type: ignore[assignment]

    normalised: list["PolicyGraduationAssessment"] = []
    for candidate in candidate_list:
        if _PolicyGraduationAssessment is not None and isinstance(
            candidate, _PolicyGraduationAssessment
        ):
            normalised.append(candidate)
            continue
        if all(
            hasattr(candidate, attribute)
            for attribute in ("policy_id", "current_stage", "recommended_stage")
        ):
            normalised.append(candidate)
    normalised.sort(key=lambda assessment: assessment.policy_id)
    return normalised


def _build_policy_graduation_panel(
    assessments: Sequence["PolicyGraduationAssessment"]
    | Mapping[str, "PolicyGraduationAssessment"],
) -> DashboardPanel:
    resolved = _normalise_graduation_assessments(assessments)
    if not resolved:
        return DashboardPanel(
            name="Policy graduation",
            status=DashboardStatus.warn,
            headline="Policy graduation assessments unavailable",
            details=(
                "No DecisionDiary evidence analysed; run the graduation evaluator to populate release posture.",
            ),
            metadata={
                "policy_graduation": {
                    "assessments": [],
                    "summary": {
                        "ready": [],
                        "live_ready": [],
                        "blocked": {},
                        "holding": [],
                        "regressing": [],
                    },
                }
            },
        )

    details: list[str] = []
    ready: list[str] = []
    live_ready: list[str] = []
    holding: list[str] = []
    blocked: dict[str, tuple[str, ...]] = {}
    regressing: list[dict[str, str]] = []

    for assessment in resolved:
        current_stage = assessment.current_stage
        recommended_stage = assessment.recommended_stage
        current_rank = _LEDGER_STAGE_ORDER[current_stage]
        recommended_rank = _LEDGER_STAGE_ORDER[recommended_stage]
        blockers = tuple(
            assessment.stage_blockers.get(recommended_stage, ())
        )
        delta = recommended_rank - current_rank

        forced_ratio = assessment.metrics.forced_ratio
        streak = assessment.metrics.consecutive_normal_latest_stage
        descriptor = (
            f"{assessment.policy_id}: {current_stage.value} -> {recommended_stage.value} "
            f"(forced {forced_ratio:.1%}, streak {streak})"
        )

        if delta < 0:
            regressing.append(
                {
                    "policy_id": assessment.policy_id,
                    "current_stage": current_stage.value,
                    "recommended_stage": recommended_stage.value,
                }
            )
            details.append(f"{descriptor} — regression")
            continue

        if delta == 0:
            holding.append(assessment.policy_id)
            details.append(f"{descriptor} — holding")
            continue

        if blockers:
            blocked[assessment.policy_id] = blockers
            blocker_text = ", ".join(blockers)
            details.append(f"{descriptor} — blocked: {blocker_text}")
            continue

        ready.append(assessment.policy_id)
        if recommended_stage is PolicyLedgerStage.LIMITED_LIVE:
            live_ready.append(assessment.policy_id)
            details.append(f"{descriptor} — limited live ready")
        else:
            details.append(f"{descriptor} — ready")

    status = DashboardStatus.ok
    if regressing:
        status = DashboardStatus.fail
    elif blocked:
        status = DashboardStatus.warn

    if regressing:
        regressing_policies = ", ".join(
            sorted(entry["policy_id"] for entry in regressing)
        )
        headline = (
            f"Regression detected for: {regressing_policies}"
            if regressing_policies
            else "Regression detected"
        )
    elif live_ready:
        headline = "Limited live ready: " + ", ".join(sorted(live_ready))
        if blocked:
            headline += f" | Blocked: {len(blocked)}"
    elif ready:
        headline = f"Promotions ready: {len(ready)}"
        if blocked:
            headline += f" | Blocked: {len(blocked)}"
    elif blocked:
        headline = f"Promotions blocked: {len(blocked)}"
    elif holding:
        headline = "All policies holding current stage"
    else:
        headline = "Policy graduation posture unavailable"

    metadata_payload = {
        "assessments": [assessment.to_dict() for assessment in resolved],
        "summary": {
            "ready": sorted(ready),
            "live_ready": sorted(live_ready),
            "blocked": {key: list(value) for key, value in blocked.items()},
            "holding": sorted(holding),
            "regressing": regressing,
        },
    }

    return DashboardPanel(
        name="Policy graduation",
        status=status,
        headline=headline,
        details=tuple(details),
        metadata={"policy_graduation": metadata_payload},
    )


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
    loop_results: Sequence["AlphaTradeLoopResult"] | None = None,
    compliance_events: Sequence["ComplianceEvent"] | None = None,
    compliance_slo_snapshot: OperationalSLOSnapshot | None = None,
    evolution_kpis: EvolutionKpiSnapshot | Mapping[str, Any] | None = None,
    operator_leverage_snapshot: OperatorLeverageSnapshot | Mapping[str, Any] | None = None,
    event_bus_snapshot: EventBusHealthSnapshot | None = None,
    monitoring_snapshot: MonitoringSnapshot | Mapping[str, Any] | None = None,
    slo_snapshot: OperationalSLOSnapshot | None = None,
    backbone_snapshot: DataBackboneReadinessSnapshot | None = None,
    operational_readiness_snapshot: OperationalReadinessSnapshot | None = None,
    policy_graduation_assessments:
        Sequence["PolicyGraduationAssessment"]
        | Mapping[str, "PolicyGraduationAssessment"]
        | None = None,
    quality_snapshot: QualityTelemetrySnapshot | None = None,
    policy_reflection: "PolicyReflectionArtifacts" | None = None,
    understanding_snapshot: UnderstandingLoopSnapshot | None = None,
    additional_panels: Sequence[DashboardPanel] | None = None,
    gate_dashboard: GateDashboard | Sequence[Any] | Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ObservabilityDashboard:
    """Compose the high-impact observability dashboard."""

    panels: list[DashboardPanel] = list(additional_panels or [])
    compliance_inputs: list[Any] = []
    if loop_results:
        for result in loop_results:
            events = getattr(result, "compliance_events", ())
            if events:
                compliance_inputs.extend(events)
    if compliance_events:
        compliance_inputs.extend(compliance_events)

    monitoring_payload: dict[str, Any] | None = None
    if isinstance(monitoring_snapshot, MonitoringSnapshot):
        monitoring_payload = monitoring_snapshot.as_dict()
    elif isinstance(monitoring_snapshot, Mapping):
        monitoring_payload = dict(monitoring_snapshot)

    gate_dashboard_obj: GateDashboard | None = None
    if gate_dashboard is not None:
        if isinstance(gate_dashboard, GateDashboard):
            gate_dashboard_obj = gate_dashboard
        elif isinstance(gate_dashboard, Mapping):
            gate_dashboard_obj = build_gate_dashboard_from_mapping(gate_dashboard)
        elif isinstance(gate_dashboard, Sequence) and not isinstance(
            gate_dashboard, (str, bytes)
        ):
            gate_dashboard_obj = build_gate_dashboard(gate_dashboard)
        else:  # pragma: no cover - defensive typing guard
            raise TypeError(
                "gate_dashboard must be a GateDashboard, mapping, or sequence of metrics"
            )

        if gate_dashboard_obj is not None and gate_dashboard_obj.generated_at.tzinfo is None:
            gate_dashboard_obj = replace(
                gate_dashboard_obj,
                generated_at=gate_dashboard_obj.generated_at.replace(tzinfo=UTC),
            )

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

    if gate_dashboard_obj is not None:
        panels.append(
            DashboardPanel(
                name="Gate Dashboard",
                status=_GATE_STATUS_TO_DASHBOARD[gate_dashboard_obj.status()],
                headline=gate_dashboard_obj.headline(),
                details=gate_dashboard_obj.panel_details(),
                metadata={"gate_dashboard": gate_dashboard_obj.as_dict()},
            )
        )

    panels.append(
        _build_compliance_panel(
            compliance_inputs,
            slo_snapshot=compliance_slo_snapshot,
            generated_at=generated_at,
        )
    )
    panels.append(_build_evolution_panel(evolution_kpis))
    panels.append(_build_operator_leverage_panel(operator_leverage_snapshot))

    if (
        event_bus_snapshot is not None
        or slo_snapshot is not None
        or monitoring_payload is not None
    ):
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

        if monitoring_payload is not None:
            latency_metadata["monitoring"] = monitoring_payload
            latency_status = _escalate(
                latency_status,
                _map_monitoring_status(monitoring_payload.get("status")),
            )

            latency_info = monitoring_payload.get("latency_ms") or monitoring_payload.get(
                "latency"
            )
            if isinstance(latency_info, Mapping) and latency_info:
                latency_status = _escalate(
                    latency_status,
                    _map_monitoring_status(latency_info.get("status")),
                )
                p95_value = latency_info.get("p95_ms") or latency_info.get("p95")
                p99_value = latency_info.get("p99_ms") or latency_info.get("p99")
                p95_text = _format_ms(p95_value)
                p99_text = _format_ms(p99_value)
                latency_line = f"Latency p95 {p95_text}, p99 {p99_text}"
                target_value = (
                    latency_info.get("p99_target_ms")
                    or latency_info.get("target_p99_ms")
                    or latency_info.get("p99_budget_ms")
                    or latency_info.get("p99_threshold_ms")
                )
                if target_value is not None:
                    latency_line += f" (target {_format_ms(target_value)})"
                sample_count = latency_info.get("samples")
                coerced_samples = _coerce_int(sample_count)
                if coerced_samples is not None:
                    latency_line += f" samples {coerced_samples}"
                details.append(latency_line)
                if _coerce_float(p99_value) is not None:
                    headline_parts.append(f"p99 {_format_ms(p99_value)}")

            throughput_info = monitoring_payload.get("throughput")
            if isinstance(throughput_info, Mapping) and throughput_info:
                latency_status = _escalate(
                    latency_status,
                    _map_monitoring_status(throughput_info.get("status")),
                )
                throughput_parts: list[str] = []
                per_minute = (
                    throughput_info.get("per_minute")
                    or throughput_info.get("per_min")
                    or throughput_info.get("orders_per_minute")
                )
                per_second = (
                    throughput_info.get("per_second")
                    or throughput_info.get("per_sec")
                    or throughput_info.get("orders_per_second")
                )
                backlog = throughput_info.get("backlog")
                per_minute_value = _coerce_float(per_minute)
                if per_minute is not None:
                    throughput_parts.append(_format_rate(per_minute, "per min"))
                if per_second is not None:
                    throughput_parts.append(_format_rate(per_second, "per sec"))
                if backlog is not None:
                    throughput_parts.append(f"backlog {int(_coerce_int(backlog) or 0)}")
                if throughput_parts:
                    details.append("Throughput " + ", ".join(throughput_parts))
                if per_minute_value is not None:
                    headline_parts.append(f"Throughput {per_minute_value:.1f}/min")

            pnl_info = monitoring_payload.get("pnl") or monitoring_payload.get("pnl_swings")
            if isinstance(pnl_info, Mapping) and pnl_info:
                latency_status = _escalate(
                    latency_status,
                    _map_monitoring_status(pnl_info.get("status")),
                )
                pnl_parts: list[str] = []
                daily = pnl_info.get("daily") or pnl_info.get("daily_pnl")
                daily_value = _coerce_float(daily)
                if daily_value is not None:
                    pnl_parts.append(f"Daily P&L {_format_currency(daily_value)}")
                    headline_parts.append(f"Daily P&L {_format_currency(daily_value)}")
                swing = (
                    pnl_info.get("swing")
                    or pnl_info.get("volatility")
                    or pnl_info.get("stdev")
                )
                swing_value = _coerce_float(swing)
                if swing_value is not None:
                    pnl_parts.append(f"σ {swing_value:.3f}")
                drawdown = pnl_info.get("drawdown") or pnl_info.get("max_drawdown")
                drawdown_value = _coerce_float(drawdown)
                if drawdown_value is not None:
                    pnl_parts.append(f"Drawdown {_format_percentage(drawdown_value)}")
                if pnl_parts:
                    details.append("P&L " + ", ".join(pnl_parts))

            memory_info = monitoring_payload.get("memory")
            if isinstance(memory_info, Mapping) and memory_info:
                latency_status = _escalate(
                    latency_status,
                    _map_monitoring_status(memory_info.get("status")),
                )
                memory_parts: list[str] = []
                current_mb = (
                    memory_info.get("current_mb")
                    or memory_info.get("current")
                    or memory_info.get("used_mb")
                )
                peak_mb = memory_info.get("peak_mb") or memory_info.get("max_mb")
                current_percent = memory_info.get("current_percent")
                peak_percent = memory_info.get("peak_percent") or memory_info.get("max_percent")
                if current_mb is not None:
                    memory_parts.append(f"current {_format_memory(current_mb)}")
                if peak_mb is not None:
                    memory_parts.append(f"peak {_format_memory(peak_mb)}")
                if current_percent is not None:
                    memory_parts.append(f"curr {_format_percentage(current_percent)}")
                if peak_percent is not None:
                    memory_parts.append(f"peak {_format_percentage(peak_percent)}")
                if memory_parts:
                    details.append("Memory " + ", ".join(memory_parts))

            tail_spikes = monitoring_payload.get("tail_spikes")
            if isinstance(tail_spikes, Sequence):
                for entry in tail_spikes:
                    if not isinstance(entry, Mapping):
                        continue
                    severity = entry.get("severity")
                    latency_status = _escalate(
                        latency_status, _map_monitoring_status(severity)
                    )
                    name = str(entry.get("name") or entry.get("metric") or "tail")
                    value = entry.get("value") or entry.get("magnitude")
                    value_numeric = _coerce_float(value)
                    unit = str(entry.get("unit") or "").strip()
                    if value_numeric is None:
                        value_text = str(value)
                    else:
                        if unit == "%" or entry.get("percent", False):
                            value_text = _format_percentage(value_numeric)
                        else:
                            value_text = (
                                f"{value_numeric:.3f}{unit}" if unit else f"{value_numeric:.3f}"
                            )
                    threshold_raw = entry.get("threshold") or entry.get("limit")
                    threshold_text = ""
                    if threshold_raw is not None:
                        threshold_numeric = _coerce_float(threshold_raw)
                        if threshold_numeric is None:
                            threshold_text = str(threshold_raw)
                        elif unit == "%" or entry.get("percent", False):
                            threshold_text = _format_percentage(threshold_numeric)
                        else:
                            threshold_text = (
                                f"{threshold_numeric:.3f}{unit}"
                                if unit
                                else f"{threshold_numeric:.3f}"
                            )
                    detail = f"Tail spike {name}: {value_text}"
                    if threshold_text:
                        detail += f" (threshold {threshold_text})"
                    if severity:
                        detail += f" [{severity}]"
                    details.append(detail)

            drift_payload = _normalise_drift_payload(
                monitoring_payload.get("drift_summary")
                or monitoring_payload.get("drift")
            )
            if drift_payload is not None:
                latency_metadata["drift"] = drift_payload
                drift_status = _map_drift_status(drift_payload.get("status"))
                latency_status = _escalate(latency_status, drift_status)
                drift_status_text = str(drift_payload.get("status") or "normal")
                headline_parts.append(f"Drift {drift_status_text}")
                dimensions = drift_payload.get("dimensions")
                if isinstance(dimensions, Mapping) and dimensions:
                    sorted_dims = sorted(
                        dimensions.items(),
                        key=lambda item: _STATUS_ORDER[
                            _map_monitoring_status(
                                item[1].get("severity") if isinstance(item[1], Mapping) else None
                            )
                        ],
                        reverse=True,
                    )
                    for name, dimension in sorted_dims[:3]:
                        if not isinstance(dimension, Mapping):
                            continue
                        severity = dimension.get("severity")
                        delta_value = _coerce_float(
                            dimension.get("delta") or dimension.get("drift_ratio")
                        )
                        variance_value = _coerce_float(dimension.get("variance_ratio"))
                        samples_value = _coerce_int(dimension.get("samples"))
                        line = f"Drift {name}: {severity or 'normal'}"
                        if delta_value is not None:
                            line += f" Δ {delta_value:.3f}"
                        if variance_value is not None:
                            line += f" σ² x{variance_value:.2f}"
                        if samples_value is not None:
                            line += f" (n={samples_value})"
                        details.append(line)
                        if _map_monitoring_status(severity) is DashboardStatus.fail:
                            headline_parts.append(f"{name} drift alert")

        if headline_parts:
            headline = "; ".join(dict.fromkeys(headline_parts))
        else:
            headline = "Latency overview"

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

    if policy_graduation_assessments is not None:
        panels.append(
            _build_policy_graduation_panel(policy_graduation_assessments)
        )

    panels.append(_build_diary_panel(loop_results))

    if understanding_snapshot is not None:
        export_understanding_throttle_metrics(understanding_snapshot)
        understanding_status = _map_understanding_status(understanding_snapshot.status)
        regime = understanding_snapshot.regime_state
        decision = understanding_snapshot.decision
        ledger_diff = understanding_snapshot.ledger_diff
        drift_exceeded = len(understanding_snapshot.drift_summary.exceeded)
        details = [
            "Regime {regime} @ {confidence:.1%}".format(
                regime=regime.regime,
                confidence=regime.confidence,
            ),
            "Drift exceedances: {count}".format(count=drift_exceeded),
            "Selected tactic {tactic} weight {weight:.3f}".format(
                tactic=decision.tactic_id,
                weight=decision.selected_weight,
            ),
            "Ledger approvals: {approvals}".format(
                approvals=",".join(ledger_diff.approvals) or "none",
            ),
        ]
        experiments = ",".join(decision.experiments_applied)
        details.append(f"Experiments: {experiments or 'none'}")

        panels.append(
            DashboardPanel(
                name="Understanding loop",
                status=understanding_status,
                headline=(
                    f"Understanding {understanding_snapshot.status.value}"
                ),
                details=tuple(details),
                metadata={
                    "understanding_loop": understanding_snapshot.as_dict()
                },
            )
        )
    else:
        panels.append(
            DashboardPanel(
                name="Understanding loop",
                status=DashboardStatus.warn,
                headline="Understanding diagnostics unavailable",
                details=(
                    "No understanding loop snapshot provided; run the graph diagnostics CLI to rebuild artifacts.",
                ),
                metadata={
                    "understanding_loop": {
                        "status": "missing",
                        "recommended_cli": "python -m tools.understanding.graph_diagnostics",
                    }
                },
            )
        )

    if policy_reflection is not None:
        panels.append(_build_policy_reflection_panel(policy_reflection))

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
