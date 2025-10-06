"""Operational service-level objective evaluation helpers.

This module translates ingest health and metrics telemetry into the
"operational readiness" artefacts called out in the roadmap.  It exposes
dataclasses that mirror the context-pack language (SLO records, alert routes,
and markdown snapshots) so runtime code can surface actionable status reports
without bespoke formatting in each caller.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Callable, Mapping

from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import (
    IngestDimensionMetrics,
    IngestMetricsSnapshot,
)
from src.operational import metrics as operational_metrics


logger = logging.getLogger(__name__)


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


def _coerce_seconds(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric < 0.0:
        return 0.0
    return numeric


def _coerce_ratio(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _record_metric(func: Callable[..., None], *args: object) -> None:
    try:
        func(*args)
    except Exception:
        logger.debug("Failed to record understanding loop metric", exc_info=True)


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


@dataclass(frozen=True)
class UnderstandingLoopSLOInputs:
    """Raw telemetry used to evaluate understanding loop SLO probes."""

    latency_seconds: float | None = None
    drift_alert_age_seconds: float | None = None
    replay_determinism_ratio: float | None = None
    generated_at: datetime | None = None
    latency_samples: int | None = None
    drift_alert_count: int | None = None
    replay_trials: int | None = None
    service: str = "understanding_loop"
    metadata: Mapping[str, object] = field(default_factory=dict)
    alert_routes: Mapping[str, str] | None = None
    latency_target_seconds: float = 0.75
    latency_warn_multiplier: float = 1.25
    latency_breach_multiplier: float = 1.5
    drift_freshness_target_seconds: float = 600.0
    drift_warn_multiplier: float = 1.5
    drift_breach_multiplier: float = 2.0
    replay_target_ratio: float = 0.995
    replay_warn_delta: float = 0.01
    replay_breach_delta: float = 0.05


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

    snapshot_metadata: dict[str, object] = {"ingest_health": health_report.status.value}
    if metadata:
        snapshot_metadata.update({str(k): v for k, v in metadata.items()})

    return OperationalSLOSnapshot(
        service="timescale_ingest",
        generated_at=generated_at,
        status=overall_status,
        slos=slos,
        metadata=snapshot_metadata,
    )


def evaluate_understanding_loop_slos(
    inputs: UnderstandingLoopSLOInputs,
) -> OperationalSLOSnapshot:
    """Evaluate understanding loop latency, drift freshness, and replay SLOs."""

    generated_at = inputs.generated_at
    if generated_at is None:
        generated_at = datetime.now(tz=UTC)
    elif generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)

    routes = (
        {str(key): str(value) for key, value in inputs.alert_routes.items() if value}
        if inputs.alert_routes
        else {}
    )

    def _route(key: str) -> str | None:
        if key in routes:
            return routes[key]
        qualified = f"{inputs.service}.{key}"
        if qualified in routes:
            return routes[qualified]
        return routes.get(inputs.service)

    latency_value = _coerce_seconds(inputs.latency_seconds)
    latency_target = max(float(inputs.latency_target_seconds), 0.0)
    latency_warn = latency_target * max(float(inputs.latency_warn_multiplier), 1.0)
    latency_breach = latency_target * max(float(inputs.latency_breach_multiplier), 1.0)

    if latency_value is None:
        latency_status = SLOStatus.at_risk
        latency_message = "Loop latency telemetry unavailable"
    elif latency_target == 0.0 and latency_value > 0.0:
        latency_status = SLOStatus.breached
        latency_message = f"Loop latency {latency_value:.3f}s recorded with zero target"
    elif latency_value > latency_breach:
        latency_status = SLOStatus.breached
        latency_message = (
            f"Loop latency {latency_value:.3f}s breaches {latency_target:.3f}s target"
        )
    elif latency_value > latency_warn:
        latency_status = SLOStatus.at_risk
        latency_message = (
            f"Loop latency {latency_value:.3f}s exceeds {latency_target:.3f}s target"
        )
    else:
        latency_status = SLOStatus.met
        latency_message = (
            f"Loop latency {latency_value:.3f}s within {latency_target:.3f}s target"
        )

    latency_observed: dict[str, object] = {}
    if latency_value is not None:
        latency_observed["latency_seconds"] = latency_value
    if inputs.latency_samples is not None:
        latency_observed["samples"] = max(int(inputs.latency_samples), 0)

    latency_metadata = {
        "warn_threshold_seconds": latency_warn,
        "breach_threshold_seconds": latency_breach,
    }

    drift_age = _coerce_seconds(inputs.drift_alert_age_seconds)
    drift_target = max(float(inputs.drift_freshness_target_seconds), 0.0)
    drift_warn = drift_target * max(float(inputs.drift_warn_multiplier), 1.0)
    drift_breach = drift_target * max(float(inputs.drift_breach_multiplier), 1.0)

    if drift_age is None:
        drift_status = SLOStatus.at_risk
        drift_message = "Drift alert freshness unknown"
    elif drift_target == 0.0 and drift_age > 0.0:
        drift_status = SLOStatus.breached
        drift_message = f"Last drift alert {drift_age:.1f}s with zero freshness target"
    elif drift_age > drift_breach:
        drift_status = SLOStatus.breached
        drift_message = (
            f"Last drift alert {drift_age:.1f}s breaches {drift_target:.1f}s freshness"
        )
    elif drift_age > drift_warn:
        drift_status = SLOStatus.at_risk
        drift_message = (
            f"Last drift alert {drift_age:.1f}s exceeds {drift_target:.1f}s freshness"
        )
    else:
        drift_status = SLOStatus.met
        drift_message = (
            f"Last drift alert {drift_age:.1f}s within {drift_target:.1f}s freshness"
        )

    drift_observed: dict[str, object] = {}
    if drift_age is not None:
        drift_observed["age_seconds"] = drift_age
    if inputs.drift_alert_count is not None:
        drift_observed["alerts"] = max(int(inputs.drift_alert_count), 0)

    drift_metadata = {
        "warn_threshold_seconds": drift_warn,
        "breach_threshold_seconds": drift_breach,
    }

    replay_ratio = _coerce_ratio(inputs.replay_determinism_ratio)
    replay_target = min(max(float(inputs.replay_target_ratio), 0.0), 1.0)
    replay_warn = max(replay_target - float(inputs.replay_warn_delta), 0.0)
    replay_breach = max(replay_target - float(inputs.replay_breach_delta), 0.0)
    if replay_breach > replay_warn:
        replay_breach, replay_warn = replay_warn, replay_breach

    if replay_ratio is None:
        replay_status = SLOStatus.at_risk
        replay_message = "Replay determinism telemetry unavailable"
    elif replay_ratio < replay_breach:
        replay_status = SLOStatus.breached
        replay_message = (
            f"Replay determinism {replay_ratio:.3f} below {replay_target:.3f} target"
        )
    elif replay_ratio < replay_warn:
        replay_status = SLOStatus.at_risk
        replay_message = (
            f"Replay determinism {replay_ratio:.3f} nearing {replay_target:.3f} target"
        )
    else:
        replay_status = SLOStatus.met
        replay_message = (
            f"Replay determinism {replay_ratio:.3f} within {replay_target:.3f} target"
        )

    replay_observed: dict[str, object] = {}
    if replay_ratio is not None:
        replay_observed["determinism_ratio"] = replay_ratio
    if inputs.replay_trials is not None:
        replay_observed["trials"] = max(int(inputs.replay_trials), 0)

    replay_metadata = {
        "warn_threshold_ratio": replay_warn,
        "breach_threshold_ratio": replay_breach,
    }

    slos: list[ServiceSLO] = [
        ServiceSLO(
            name=f"{inputs.service}.latency",
            status=latency_status,
            message=latency_message,
            target={"p95_seconds": latency_target},
            observed=latency_observed,
            alert_route=_route("latency"),
            metadata=latency_metadata,
        ),
        ServiceSLO(
            name=f"{inputs.service}.drift_freshness",
            status=drift_status,
            message=drift_message,
            target={"freshness_seconds": drift_target},
            observed=drift_observed,
            alert_route=_route("drift_freshness"),
            metadata=drift_metadata,
        ),
        ServiceSLO(
            name=f"{inputs.service}.replay_determinism",
            status=replay_status,
            message=replay_message,
            target={"determinism_ratio": replay_target},
            observed=replay_observed,
            alert_route=_route("replay_determinism"),
            metadata=replay_metadata,
        ),
    ]

    overall_status = SLOStatus.met
    for slo in slos:
        overall_status = _escalate(overall_status, slo.status)

    counter = Counter(slo.status for slo in slos)
    if counter[SLOStatus.breached]:
        summary_message = f"{counter[SLOStatus.breached]} loop SLOs breached"
    elif counter[SLOStatus.at_risk]:
        summary_message = f"{counter[SLOStatus.at_risk]} loop SLOs at risk"
    else:
        summary_message = "All loop SLOs met"

    summary_metadata = {
        "status_counts": {status.value: counter.get(status, 0) for status in SLOStatus},
        "latency_status": latency_status.value,
        "drift_status": drift_status.value,
        "replay_status": replay_status.value,
    }

    summary_observed = {
        "latency_seconds": latency_value,
        "drift_age_seconds": drift_age,
        "replay_determinism_ratio": replay_ratio,
    }

    summary_slo = ServiceSLO(
        name=inputs.service,
        status=overall_status,
        message=summary_message,
        target={},
        observed=summary_observed,
        alert_route=_route(inputs.service),
        metadata=summary_metadata,
    )

    snapshot_metadata: dict[str, object] = {
        "service": inputs.service,
        "latency_status": latency_status.value,
        "drift_status": drift_status.value,
        "replay_status": replay_status.value,
    }
    if inputs.metadata:
        snapshot_metadata.update({str(key): value for key, value in inputs.metadata.items()})

    # Record metrics for Prometheus exporters; best-effort only
    _record_metric(
        operational_metrics.set_understanding_loop_latency,
        latency_value,
    )
    _record_metric(
        operational_metrics.set_understanding_loop_latency_status,
        latency_status.value,
    )
    _record_metric(
        operational_metrics.set_understanding_loop_drift_freshness,
        drift_age,
    )
    _record_metric(
        operational_metrics.set_understanding_loop_drift_status,
        drift_status.value,
    )
    _record_metric(
        operational_metrics.set_understanding_loop_replay_determinism,
        replay_ratio,
    )
    _record_metric(
        operational_metrics.set_understanding_loop_replay_status,
        replay_status.value,
    )

    slos_with_summary = tuple([summary_slo, *slos])

    return OperationalSLOSnapshot(
        service=inputs.service,
        generated_at=generated_at,
        status=overall_status,
        slos=slos_with_summary,
        metadata=snapshot_metadata,
    )


__all__ = [
    "DEFAULT_ALERT_ROUTES",
    "OperationalSLOSnapshot",
    "ServiceSLO",
    "SLOStatus",
    "UnderstandingLoopSLOInputs",
    "evaluate_ingest_slos",
    "evaluate_understanding_loop_slos",
]
