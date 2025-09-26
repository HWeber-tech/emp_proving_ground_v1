"""Composable ingest observability snapshots for runtime and CI telemetry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.coercion import coerce_float, coerce_int

from .failover import IngestFailoverDecision
from .health import IngestHealthCheck, IngestHealthReport, IngestHealthStatus
from .metrics import IngestMetricsSnapshot
from .recovery import IngestRecoveryRecommendation


def _normalise_mapping(mapping: Mapping[str, object] | None) -> dict[str, object]:
    if mapping is None:
        return {}
    return {str(key): value for key, value in mapping.items()}


def _normalise_metadata(mapping: Mapping[str, object] | None) -> dict[str, object]:
    if not mapping:
        return {}
    return _normalise_mapping(mapping)


def _coerce_optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    return None


def _coerce_symbols(value: object | None, *, fallback: Iterable[str] = ()) -> tuple[str, ...]:
    sequence: Iterable[object]
    if isinstance(value, Mapping):
        sequence = value.values()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        sequence = value
    else:
        sequence = fallback
    symbols: list[str] = []
    for symbol in sequence:
        if symbol is None:
            continue
        candidate = str(symbol).strip()
        if candidate:
            symbols.append(candidate)
    return tuple(symbols)


def _iter_metric_records(payload: object | None) -> list[dict[str, object]]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [_normalise_mapping(payload)]
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        records: list[dict[str, object]] = []
        for item in payload:
            if isinstance(item, Mapping):
                records.append(_normalise_mapping(item))
        return records
    return []


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
            reasons_payload = summary.get("reasons")
            if isinstance(reasons_payload, Mapping):
                formatted_reasons = "; ".join(
                    f"{str(dimension)}: {str(message)}"
                    for dimension, message in reasons_payload.items()
                    if dimension and message
                )
                if formatted_reasons:
                    lines.append(f"- Recovery plan reasons: {formatted_reasons}")
            plan_summary = summary.get("plan")
            if isinstance(plan_summary, Mapping):
                planned = ", ".join(
                    sorted(str(dimension) for dimension in plan_summary.keys() if dimension)
                )
                if planned:
                    lines.append(f"- Recovery plan dimensions: {planned}")
            missing_symbols = summary.get("missing_symbols")
            if isinstance(missing_symbols, Mapping):
                formatted_missing = "; ".join(
                    f"{str(dimension)}: {', '.join(sorted(_coerce_symbols(symbols)))}"
                    for dimension, symbols in missing_symbols.items()
                    if symbols
                )
                if formatted_missing:
                    lines.append(f"- Missing symbols: {formatted_missing}")
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
        merged.update(_normalise_mapping(payload))
    return merged


def _dimension_from_health(
    check: IngestHealthCheck,
    metric: Mapping[str, object] | None,
) -> IngestObservabilityDimension:
    rows = check.rows_written
    freshness = check.freshness_seconds
    duration = check.ingest_duration_seconds
    observed = tuple(check.observed_symbols)
    source: str | None = None
    if metric is not None:
        rows = coerce_int(metric.get("rows"), default=rows)
        freshness_candidate = coerce_float(metric.get("freshness_seconds"))
        if freshness_candidate is not None:
            freshness = freshness_candidate
        duration_candidate = coerce_float(metric.get("ingest_duration_seconds"))
        if duration_candidate is not None:
            duration = duration_candidate
        observed = _coerce_symbols(metric.get("symbols"), fallback=observed)
        source = _coerce_optional_str(metric.get("source"))
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
    dimension_obj = metric.get("dimension")
    dimension = str(dimension_obj) if dimension_obj is not None else "unknown"
    return IngestObservabilityDimension(
        dimension=dimension,
        status=IngestHealthStatus.ok,
        rows=coerce_int(metric.get("rows"), default=0),
        freshness_seconds=coerce_float(metric.get("freshness_seconds")),
        message="Metric recorded without health check",
        observed_symbols=_coerce_symbols(metric.get("symbols")),
        ingest_duration_seconds=coerce_float(metric.get("ingest_duration_seconds")),
        source=_coerce_optional_str(metric.get("source")),
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
    raw_dimensions = metrics_payload.get("dimensions")
    metric_records = _iter_metric_records(raw_dimensions)

    metrics_by_dimension: dict[str, Mapping[str, object]] = {}
    for record in metric_records:
        name_obj = record.get("dimension")
        if not name_obj:
            continue
        name = str(name_obj)
        if not name:
            continue
        metrics_by_dimension[name] = record

    dimensions: list[IngestObservabilityDimension] = []
    seen: set[str] = set()

    for check in health.checks:
        metric_payload = metrics_by_dimension.get(check.dimension)
        dimensions.append(_dimension_from_health(check, metric_payload))
        seen.add(check.dimension)

    for metric in metric_records:
        name_obj = metric.get("dimension")
        if not name_obj:
            continue
        name = str(name_obj)
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
