"""Operational service-level objective evaluation helpers.

This module translates ingest health and metrics telemetry into the
"operational readiness" artefacts called out in the roadmap.  It exposes
dataclasses that mirror the context-pack language (SLO records, alert routes,
and markdown snapshots) so runtime code can surface actionable status reports
without bespoke formatting in each caller.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import (
    IngestDimensionMetrics,
    IngestMetricsSnapshot,
)


class SLOStatus(StrEnum):
    """Severity levels for operational SLO evaluation."""

    met = "pass"
    at_risk = "warn"
    breached = "fail"


_SLO_ORDER: Mapping[SLOStatus, int] = {
    SLOStatus.met: 0,
    SLOStatus.at_risk: 1,
    SLOStatus.breached: 2,
}


DEFAULT_ALERT_ROUTES: dict[str, str] = {
    "timescale_ingest": "pagerduty:data-backbone",
    "timescale_ingest.daily_bars": "slack:#ops-timescale",
    "timescale_ingest.intraday_trades": "slack:#ops-timescale",
    "timescale_ingest.macro_events": "slack:#ops-timescale",
}


def _map_health_status(status: IngestHealthStatus) -> SLOStatus:
    if status is IngestHealthStatus.ok:
        return SLOStatus.met
    if status is IngestHealthStatus.warn:
        return SLOStatus.at_risk
    return SLOStatus.breached


def _escalate(current: SLOStatus, candidate: SLOStatus) -> SLOStatus:
    if _SLO_ORDER[candidate] > _SLO_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class ServiceSLO:
    """Structured representation of a single SLO evaluation."""

    name: str
    status: SLOStatus
    message: str
    target: dict[str, object] = field(default_factory=dict)
    observed: dict[str, object] = field(default_factory=dict)
    alert_route: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "target": dict(self.target),
            "observed": dict(self.observed),
        }
        if self.alert_route:
            payload["alert_route"] = self.alert_route
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class OperationalSLOSnapshot:
    """Aggregate operational SLO snapshot for a service."""

    service: str
    generated_at: datetime
    status: SLOStatus
    slos: tuple[ServiceSLO, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "slos": [slo.as_dict() for slo in self.slos],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        header = (
            f"**Operational SLOs – {self.service}** (status: {self.status.value}, "
            f"generated: {self.generated_at.isoformat()})"
        )
        lines = [header, "", "| SLO | Status | Message | Alert |", "| --- | --- | --- | --- |"]
        for slo in self.slos:
            alert = slo.alert_route or ""
            lines.append(f"| {slo.name} | {slo.status.value} | {slo.message} | {alert} |")
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)


def _dimension_metric_lookup(
    metrics: IngestMetricsSnapshot | None,
) -> dict[str, IngestDimensionMetrics]:
    if metrics is None:
        return {}
    return {metric.dimension: metric for metric in metrics.dimensions}


def _resolve_alert(
    key: str,
    *,
    routes: Mapping[str, str],
) -> str | None:
    if key in routes:
        return routes[key]
    parent = "timescale_ingest"
    return routes.get(parent)


def evaluate_ingest_slos(
    metrics: IngestMetricsSnapshot | None,
    health_report: IngestHealthReport,
    *,
    alert_routes: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> OperationalSLOSnapshot:
    """Derive operational SLO status from ingest telemetry."""

    routes: dict[str, str] = dict(DEFAULT_ALERT_ROUTES)
    if alert_routes:
        routes.update({str(k): str(v) for k, v in alert_routes.items() if v})

    generated_at = health_report.generated_at or datetime.now(tz=UTC)
    metrics_lookup = _dimension_metric_lookup(metrics)

    slo_records: list[ServiceSLO] = []
    overall_status = SLOStatus.met

    for check in health_report.checks:
        slo_status = _map_health_status(check.status)
        overall_status = _escalate(overall_status, slo_status)

        observed: dict[str, object] = {
            "rows": check.rows_written,
            "freshness_seconds": check.freshness_seconds,
        }
        metric = metrics_lookup.get(check.dimension)
        if metric is not None:
            observed.update(
                {
                    "rows": metric.rows,
                    "symbols": list(metric.symbols),
                    "freshness_seconds": metric.freshness_seconds,
                    "ingest_duration_seconds": metric.ingest_duration_seconds,
                }
            )
        elif check.observed_symbols:
            observed["symbols"] = list(check.observed_symbols)
        if check.ingest_duration_seconds is not None:
            observed.setdefault("ingest_duration_seconds", check.ingest_duration_seconds)

        target: dict[str, object] = {}
        if "freshness_sla_seconds" in check.metadata:
            target["freshness_seconds"] = check.metadata["freshness_sla_seconds"]
        if "min_rows_required" in check.metadata:
            target["min_rows"] = check.metadata["min_rows_required"]

        record_metadata: dict[str, object] = {}
        if check.missing_symbols:
            record_metadata["missing_symbols"] = list(check.missing_symbols)
        if check.expected_symbols:
            record_metadata["expected_symbols"] = list(check.expected_symbols)

        slo_records.append(
            ServiceSLO(
                name=f"timescale_ingest.{check.dimension}",
                status=slo_status,
                message=check.message,
                target=target,
                observed=observed,
                alert_route=_resolve_alert(f"timescale_ingest.{check.dimension}", routes=routes),
                metadata=record_metadata,
            )
        )

    counter = Counter(record.status for record in slo_records)
    if counter[SLOStatus.breached]:
        summary_message = f"{counter[SLOStatus.breached]} ingest SLOs breached"
    elif counter[SLOStatus.at_risk]:
        summary_message = f"{counter[SLOStatus.at_risk]} ingest SLOs at risk"
    elif slo_records:
        summary_message = "All ingest SLOs met"
    else:
        summary_message = "No ingest checks executed"

    overall_observed: dict[str, object] = {
        "health_status": health_report.status.value,
        "dimensions": [check.dimension for check in health_report.checks],
    }
    if metrics is not None:
        overall_observed["total_rows"] = metrics.total_rows()
        overall_observed["metrics_generated_at"] = metrics.generated_at.isoformat()

    summary_record = ServiceSLO(
        name="timescale_ingest",
        status=overall_status,
        message=summary_message,
        target={},
        observed=overall_observed,
        alert_route=_resolve_alert("timescale_ingest", routes=routes),
        metadata={"status_counts": {status.value: count for status, count in counter.items()}},
    )

    slos = tuple([summary_record, *slo_records])

    snapshot_metadata = {"ingest_health": health_report.status.value}
    if metadata:
        snapshot_metadata.update({str(k): v for k, v in metadata.items()})

    return OperationalSLOSnapshot(
        service="timescale_ingest",
        generated_at=generated_at,
        status=overall_status,
        slos=slos,
        metadata=snapshot_metadata,
    )


__all__ = [
    "DEFAULT_ALERT_ROUTES",
    "OperationalSLOSnapshot",
    "ServiceSLO",
    "SLOStatus",
    "evaluate_ingest_slos",
]
