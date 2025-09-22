"""Composable ingest observability snapshots for runtime and CI telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Mapping

from .failover import IngestFailoverDecision
from .health import IngestHealthCheck, IngestHealthReport, IngestHealthStatus
from .metrics import IngestMetricsSnapshot
from .recovery import IngestRecoveryRecommendation


def _normalise_metadata(mapping: Mapping[str, object] | None) -> dict[str, object]:
    if not mapping:
        return {}
    return {str(key): value for key, value in mapping.items()}


@dataclass(frozen=True)
class IngestObservabilityDimension:
    """Union of metrics and health data for a single ingest dimension."""

    dimension: str
    status: IngestHealthStatus
    rows: int
    freshness_seconds: float | None
    message: str
    observed_symbols: tuple[str, ...] = field(default_factory=tuple)
    expected_symbols: tuple[str, ...] = field(default_factory=tuple)
    missing_symbols: tuple[str, ...] = field(default_factory=tuple)
    ingest_duration_seconds: float | None = None
    source: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "status": self.status.value,
            "rows": self.rows,
            "freshness_seconds": self.freshness_seconds,
            "message": self.message,
            "observed_symbols": list(self.observed_symbols),
        }
        if self.expected_symbols:
            payload["expected_symbols"] = list(self.expected_symbols)
        if self.missing_symbols:
            payload["missing_symbols"] = list(self.missing_symbols)
        if self.ingest_duration_seconds is not None:
            payload["ingest_duration_seconds"] = self.ingest_duration_seconds
        if self.source:
            payload["source"] = self.source
        extra = _normalise_metadata(self.metadata)
        if extra:
            payload["metadata"] = extra
        return payload


@dataclass(frozen=True)
class IngestObservabilitySnapshot:
    """Aggregate observability signal for a Timescale ingest execution."""

    generated_at: datetime
    status: IngestHealthStatus
    dimensions: tuple[IngestObservabilityDimension, ...]
    metrics: IngestMetricsSnapshot
    failover: IngestFailoverDecision | None = None
    recovery: IngestRecoveryRecommendation | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def total_rows(self) -> int:
        return self.metrics.total_rows()

    def degraded_dimensions(self) -> tuple[str, ...]:
        return tuple(
            dimension.dimension
            for dimension in self.dimensions
            if dimension.status is not IngestHealthStatus.ok
        )

    def recovery_summary(self) -> dict[str, object] | None:
        if self.recovery is None:
            return None
        summary = self.recovery.summary()
        return summary or None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "total_rows": self.total_rows(),
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            "metrics": self.metrics.as_dict(),
            "degraded_dimensions": list(self.degraded_dimensions()),
            "metadata": _normalise_metadata(self.metadata),
        }
        if self.failover is not None:
            payload["failover"] = self.failover.as_dict()
        summary = self.recovery_summary()
        if summary is not None:
            payload["recovery"] = summary
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"### Timescale ingest observability – {self.generated_at.astimezone(UTC).isoformat()}",
            "",
            f"- Overall status: **{self.status.value.upper()}**",
            f"- Total rows: `{self.total_rows()}` across {len(self.dimensions)} dimensions",
        ]
        if self.failover is not None:
            flag = "YES" if self.failover.should_failover else "NO"
            detail = f" (reason: {self.failover.reason})" if self.failover.reason else ""
            lines.append(f"- Failover triggered: **{flag}**{detail}")
        summary = self.recovery_summary()
        if summary is not None:
            reasons = summary.get("reasons") or {}
            formatted_reasons = "; ".join(f"{dim}: {msg}" for dim, msg in reasons.items())
            if formatted_reasons:
                lines.append(f"- Recovery plan reasons: {formatted_reasons}")
            plan_summary = summary.get("plan")
            if plan_summary:
                planned = ", ".join(sorted(plan_summary.keys()))
                lines.append(f"- Recovery plan dimensions: {planned}")
        lines.extend(
            [
                "",
                "| Dimension | Status | Rows | Freshness (s) | Missing symbols | Source |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for dimension in self.dimensions:
            freshness = (
                "" if dimension.freshness_seconds is None else f"{dimension.freshness_seconds:.0f}"
            )
            missing = ", ".join(dimension.missing_symbols)
            source = dimension.source or ""
            lines.append(
                "| {dimension} | {status} | {rows} | {freshness} | {missing} | {source} |".format(
                    dimension=dimension.dimension,
                    status=dimension.status.value,
                    rows=dimension.rows,
                    freshness=freshness,
                    missing=missing,
                    source=source,
                )
            )
        return "\n".join(lines)


def _merge_metadata(*payloads: Mapping[str, object] | None) -> dict[str, object]:
    merged: dict[str, object] = {}
    for payload in payloads:
        if not payload:
            continue
        merged.update({str(key): value for key, value in payload.items()})
    return merged


def _dimension_from_health(
    check: IngestHealthCheck,
    metric: Mapping[str, object] | None,
) -> IngestObservabilityDimension:
    rows = check.rows_written
    freshness = check.freshness_seconds
    duration = check.ingest_duration_seconds
    observed = tuple(check.observed_symbols)
    source = None
    if metric is not None:
        rows = int(metric.get("rows", rows))
        freshness = metric.get("freshness_seconds", freshness)
        duration = metric.get("ingest_duration_seconds", duration)
        observed = tuple(metric.get("symbols", observed))
        source = metric.get("source")
    return IngestObservabilityDimension(
        dimension=check.dimension,
        status=check.status,
        rows=rows,
        freshness_seconds=freshness,
        message=check.message,
        observed_symbols=observed,
        expected_symbols=tuple(check.expected_symbols),
        missing_symbols=tuple(check.missing_symbols),
        ingest_duration_seconds=duration,
        source=source,
        metadata=check.metadata,
    )


def _dimension_from_metric(metric: Mapping[str, object]) -> IngestObservabilityDimension:
    return IngestObservabilityDimension(
        dimension=str(metric.get("dimension")),
        status=IngestHealthStatus.ok,
        rows=int(metric.get("rows", 0)),
        freshness_seconds=metric.get("freshness_seconds"),
        message="Metric recorded without health check",
        observed_symbols=tuple(metric.get("symbols", tuple())),
        ingest_duration_seconds=metric.get("ingest_duration_seconds"),
        source=metric.get("source"),
    )


def build_ingest_observability_snapshot(
    metrics: IngestMetricsSnapshot,
    health: IngestHealthReport,
    *,
    failover: IngestFailoverDecision | None = None,
    recovery: IngestRecoveryRecommendation | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IngestObservabilitySnapshot:
    """Merge ingest metrics, health, and recovery/failover metadata."""

    metrics_payload = metrics.as_dict()
    metrics_by_dimension = {
        metric["dimension"]: metric
        for metric in metrics_payload.get("dimensions", [])
        if metric.get("dimension")
    }

    dimensions: list[IngestObservabilityDimension] = []
    seen: set[str] = set()

    for check in health.checks:
        metric_payload = metrics_by_dimension.get(check.dimension)
        dimensions.append(_dimension_from_health(check, metric_payload))
        seen.add(check.dimension)

    for metric in metrics_payload.get("dimensions", []):
        name = metric.get("dimension")
        if not name or name in seen:
            continue
        dimensions.append(_dimension_from_metric(metric))

    dimensions.sort(key=lambda item: item.dimension)

    merged_metadata = _merge_metadata(health.metadata, metadata)

    active_recovery = recovery if (recovery and not recovery.plan.is_empty()) else None

    return IngestObservabilitySnapshot(
        generated_at=health.generated_at,
        status=health.status,
        dimensions=tuple(dimensions),
        metrics=metrics,
        failover=failover,
        recovery=active_recovery,
        metadata=merged_metadata,
    )


__all__ = [
    "IngestObservabilityDimension",
    "IngestObservabilitySnapshot",
    "build_ingest_observability_snapshot",
]
