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
from typing import Mapping, Sequence

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
class LoopLatencyProbe:
    """Observed latency metrics for a single understanding-loop cycle."""

    loop: str
    target_p95_seconds: float
    p95_seconds: float | None
    max_seconds: float | None
    breach_p95_seconds: float | None = None
    sample_count: int | None = None
    window_seconds: float | None = None
    runbook: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftAlertFreshnessProbe:
    """Freshness posture for sensory drift alerting."""

    alert: str
    warn_after_seconds: float
    fail_after_seconds: float
    last_alert_at: datetime | None
    alerts_sent: int = 0
    runbook: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayDeterminismProbe:
    """Determinism checks for recorded sensory replays."""

    probe: str
    warn_threshold: float
    fail_threshold: float
    drift_score: float | None
    checksum_match: bool | None = None
    mismatched_fields: tuple[str, ...] = ()
    runbook: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


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
    *,
    loop_latency_probes: Sequence[LoopLatencyProbe] | None = None,
    drift_alert_probes: Sequence[DriftAlertFreshnessProbe] | None = None,
    replay_probes: Sequence[ReplayDeterminismProbe] | None = None,
    generated_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> OperationalSLOSnapshot:
    """Grade understanding-loop SLO probes and export Prometheus gauges."""

    probes_latency: Sequence[LoopLatencyProbe] = loop_latency_probes or ()
    probes_drift: Sequence[DriftAlertFreshnessProbe] = drift_alert_probes or ()
    probes_replay: Sequence[ReplayDeterminismProbe] = replay_probes or ()

    evaluation_moment = now or datetime.now(tz=UTC)
    snapshot_moment = generated_at or evaluation_moment

    slo_records: list[ServiceSLO] = []
    overall_status = SLOStatus.met

    def _escalate_overall(status: SLOStatus) -> None:
        nonlocal overall_status
        overall_status = _escalate(overall_status, status)

    for probe in probes_latency:
        loop_name = probe.loop or "loop"
        target = max(float(probe.target_p95_seconds), 0.0)
        breach = (
            float(probe.breach_p95_seconds)
            if probe.breach_p95_seconds is not None
            else target * 1.5 if target else 0.0
        )
        p95 = float(probe.p95_seconds) if probe.p95_seconds is not None else None
        max_latency = float(probe.max_seconds) if probe.max_seconds is not None else None

        status = SLOStatus.met
        if p95 is None:
            status = SLOStatus.breached
            message = "Missing loop latency telemetry"
        else:
            if p95 > target:
                status = SLOStatus.at_risk if p95 <= breach else SLOStatus.breached
            if max_latency is not None and max_latency > breach:
                status = SLOStatus.breached
            if status is SLOStatus.met:
                message = f"p95 {p95:.2f}s within target {target:.2f}s"
            elif status is SLOStatus.at_risk:
                message = f"p95 {p95:.2f}s above target {target:.2f}s"
            else:
                message = f"p95 {p95:.2f}s exceeds breach threshold {breach:.2f}s"

        observed: dict[str, object] = {}
        if p95 is not None:
            observed["p95_seconds"] = p95
        if max_latency is not None:
            observed["max_seconds"] = max_latency
        if probe.sample_count is not None:
            observed["samples"] = int(probe.sample_count)
        if probe.window_seconds is not None:
            observed["window_seconds"] = float(probe.window_seconds)

        target_payload: dict[str, object] = {"p95_seconds": target}
        if breach:
            target_payload["p95_fail_seconds"] = breach

        record_metadata: dict[str, object] = dict(probe.metadata)
        if probe.runbook:
            record_metadata.setdefault("runbook", probe.runbook)

        slo_records.append(
            ServiceSLO(
                name=f"understanding_loop.latency.{loop_name}",
                status=status,
                message=message,
                target=target_payload,
                observed=observed,
                metadata=record_metadata,
            )
        )

        if p95 is not None:
            operational_metrics.set_understanding_loop_latency(loop_name, "p95", p95)
        if max_latency is not None:
            operational_metrics.set_understanding_loop_latency(loop_name, "max", max_latency)
        operational_metrics.set_understanding_loop_latency_status(
            loop_name, _SLO_ORDER[status]
        )
        _escalate_overall(status)

    for probe in probes_drift:
        alert = probe.alert or "drift_alert"
        warn_after = max(float(probe.warn_after_seconds), 0.0)
        fail_after = max(float(probe.fail_after_seconds), warn_after)
        last_alert = probe.last_alert_at
        if last_alert is not None and last_alert.tzinfo is None:
            last_alert = last_alert.replace(tzinfo=UTC)
        freshness: float | None
        if last_alert is None:
            freshness = None
        else:
            freshness = max(
                (evaluation_moment - last_alert.astimezone(UTC)).total_seconds(),
                0.0,
            )

        if freshness is None:
            status = SLOStatus.breached
            message = "No drift alert observed"
        elif freshness <= warn_after:
            status = SLOStatus.met
            message = f"Freshness {freshness:.0f}s within {warn_after:.0f}s target"
        elif freshness <= fail_after:
            status = SLOStatus.at_risk
            message = f"Freshness {freshness:.0f}s approaching limit {fail_after:.0f}s"
        else:
            status = SLOStatus.breached
            message = f"Freshness {freshness:.0f}s exceeds limit {fail_after:.0f}s"

        observed = {
            "alerts_sent": int(probe.alerts_sent),
        }
        if freshness is not None:
            observed["freshness_seconds"] = freshness
        if last_alert is not None:
            observed["last_alert_at"] = last_alert.astimezone(UTC).isoformat()

        target_payload = {
            "freshness_warn_seconds": warn_after,
            "freshness_fail_seconds": fail_after,
        }

        record_metadata: dict[str, object] = dict(probe.metadata)
        if probe.runbook:
            record_metadata.setdefault("runbook", probe.runbook)

        slo_records.append(
            ServiceSLO(
                name=f"understanding_loop.drift_alert.{alert}",
                status=status,
                message=message,
                target=target_payload,
                observed=observed,
                metadata=record_metadata,
            )
        )

        operational_metrics.set_drift_alert_freshness(alert, freshness)
        operational_metrics.set_drift_alert_status(alert, _SLO_ORDER[status])
        _escalate_overall(status)

    for probe in probes_replay:
        probe_name = probe.probe or "replay"
        warn_threshold = float(probe.warn_threshold)
        fail_threshold = max(float(probe.fail_threshold), warn_threshold)
        drift = float(probe.drift_score) if probe.drift_score is not None else None

        status = SLOStatus.met
        issues: list[str] = []
        checksum_match = probe.checksum_match
        if checksum_match is False:
            status = SLOStatus.breached
            issues.append("checksum mismatch")
        elif checksum_match is None:
            status = SLOStatus.at_risk
            issues.append("checksum unknown")

        if drift is None:
            status = SLOStatus.breached
            issues.append("missing drift telemetry")
        else:
            if drift > fail_threshold:
                status = SLOStatus.breached
                issues.append(f"drift {drift:.3f} exceeds {fail_threshold:.3f}")
            elif drift > warn_threshold and status is not SLOStatus.breached:
                status = _escalate(status, SLOStatus.at_risk)
                issues.append(f"drift {drift:.3f} above warn {warn_threshold:.3f}")

        if probe.mismatched_fields:
            status = SLOStatus.breached
            issues.append("mismatched fields present")

        if not issues:
            message = "Replay determinism within thresholds"
        else:
            message = "; ".join(issues)

        observed: dict[str, object] = {
            "checksum_match": checksum_match,
            "mismatched_fields": list(probe.mismatched_fields),
        }
        if drift is not None:
            observed["drift_score"] = drift

        target_payload = {
            "warn_threshold": warn_threshold,
            "fail_threshold": fail_threshold,
        }

        record_metadata: dict[str, object] = dict(probe.metadata)
        if probe.runbook:
            record_metadata.setdefault("runbook", probe.runbook)

        slo_records.append(
            ServiceSLO(
                name=f"understanding_loop.replay.{probe_name}",
                status=status,
                message=message,
                target=target_payload,
                observed=observed,
                metadata=record_metadata,
            )
        )

        operational_metrics.set_replay_determinism_drift(probe_name, drift)
        operational_metrics.set_replay_determinism_status(probe_name, _SLO_ORDER[status])
        operational_metrics.set_replay_determinism_mismatches(
            probe_name, len(probe.mismatched_fields)
        )
        _escalate_overall(status)

    detail_records = tuple(slo_records)
    counter = Counter(record.status for record in detail_records)
    if counter[SLOStatus.breached]:
        summary_message = f"{counter[SLOStatus.breached]} loop SLOs breached"
    elif counter[SLOStatus.at_risk]:
        summary_message = f"{counter[SLOStatus.at_risk]} loop SLOs at risk"
    elif detail_records:
        summary_message = "All understanding-loop SLOs met"
    else:
        summary_message = "No understanding-loop probes evaluated"

    summary_observed: dict[str, object] = {
        "loop_latency_probes": len(probes_latency),
        "drift_alert_probes": len(probes_drift),
        "replay_probes": len(probes_replay),
    }

    summary_record = ServiceSLO(
        name="understanding_loop",
        status=overall_status,
        message=summary_message,
        target={},
        observed=summary_observed,
        metadata={"status_counts": {status.value: count for status, count in counter.items()}},
    )

    snapshot_metadata: dict[str, object] = {}
    if metadata:
        snapshot_metadata.update({str(k): v for k, v in metadata.items()})

    return OperationalSLOSnapshot(
        service="understanding_loop",
        generated_at=snapshot_moment,
        status=overall_status,
        slos=tuple([summary_record, *detail_records]),
        metadata=snapshot_metadata,
    )


__all__ = [
    "DEFAULT_ALERT_ROUTES",
    "DriftAlertFreshnessProbe",
    "LoopLatencyProbe",
    "OperationalSLOSnapshot",
    "ReplayDeterminismProbe",
    "ServiceSLO",
    "SLOStatus",
    "evaluate_ingest_slos",
    "evaluate_understanding_loop_slos",
]
