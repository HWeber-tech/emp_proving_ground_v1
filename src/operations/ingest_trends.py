"""Ingest trend evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from src.core.event_bus import Event, EventBus
from src.data_foundation.persist.timescale import TimescaleIngestRunRecord
from src.operations.event_bus_failover import publish_event_with_failover
import src.operational.metrics as operational_metrics


logger = logging.getLogger(__name__)


class IngestTrendStatus(StrEnum):
    """Severity levels describing ingest trend posture."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[IngestTrendStatus, int] = {
    IngestTrendStatus.ok: 0,
    IngestTrendStatus.warn: 1,
    IngestTrendStatus.fail: 2,
}


def _escalate(current: IngestTrendStatus, candidate: IngestTrendStatus) -> IngestTrendStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_status(value: str | None) -> IngestTrendStatus:
    if value is None:
        return IngestTrendStatus.warn

    lowered = str(value).strip().lower()
    if lowered in {"ok", "success", "skipped"}:
        return IngestTrendStatus.ok
    if lowered in {"warn", "warning", "degraded", "degrade"}:
        return IngestTrendStatus.warn
    if lowered in {"fail", "failed", "error"}:
        return IngestTrendStatus.fail
    return IngestTrendStatus.warn


@dataclass(frozen=True)
class IngestDimensionTrend:
    """Trend information for a specific ingest dimension."""

    dimension: str
    status: IngestTrendStatus
    statuses: tuple[IngestTrendStatus, ...]
    rows_written: tuple[int, ...]
    freshness_seconds: tuple[float | None, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "status": self.status.value,
            "statuses": [status.value for status in self.statuses],
            "rows_written": list(self.rows_written),
            "freshness_seconds": list(self.freshness_seconds),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class IngestTrendSnapshot:
    """Aggregated ingest trend snapshot."""

    generated_at: datetime
    status: IngestTrendStatus
    lookback: int
    dimensions: tuple[IngestDimensionTrend, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "lookback": self.lookback,
            "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        lines = [
            f"**Ingest trends**",
            f"- Status: {self.status.value}",
            f"- Generated: {self.generated_at.astimezone(UTC).isoformat()}",
            f"- Lookback: last {self.lookback} runs",
        ]
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        for dimension in self.dimensions:
            lines.append("")
            lines.append(f"### {dimension.dimension}")
            lines.append(f"- Status: {dimension.status.value}")
            if dimension.issues:
                lines.append("- Issues:")
                for issue in dimension.issues:
                    lines.append(f"  - {issue}")
            if dimension.rows_written:
                lines.append(
                    "- Rows written: " + ", ".join(str(value) for value in dimension.rows_written)
                )
            if dimension.freshness_seconds:
                freshness_display = [
                    "∅" if value is None else f"{value:.1f}s"
                    for value in dimension.freshness_seconds
                ]
                lines.append("- Freshness: " + ", ".join(freshness_display))
        return "\n".join(lines)


def format_ingest_trends_markdown(snapshot: IngestTrendSnapshot) -> str:
    """Convenience wrapper mirroring other telemetry helpers."""

    return snapshot.to_markdown()


def _build_dimension_metadata(records: Sequence[TimescaleIngestRunRecord]) -> dict[str, object]:
    recent_runs = [
        {
            "run_id": record.run_id,
            "status": _normalise_status(record.status).value,
            "rows_written": record.rows_written,
            "freshness_seconds": record.freshness_seconds,
            "executed_at": record.executed_at.astimezone(UTC).isoformat(),
            "source": record.source,
            "symbols": list(record.symbols),
        }
        for record in records
    ]
    return {"recent_runs": recent_runs}


def _compute_rows_issue(rows: Sequence[int]) -> tuple[str | None, IngestTrendStatus]:
    if len(rows) < 2:
        return None, IngestTrendStatus.ok

    latest = rows[0]
    previous = [value for value in rows[1:] if value is not None]
    if not previous:
        return None, IngestTrendStatus.ok

    average_previous = mean(previous)
    if average_previous <= 0:
        return None, IngestTrendStatus.ok

    if latest == 0 and average_previous > 0:
        return "Latest run wrote 0 rows after recent non-zero averages", IngestTrendStatus.warn

    if latest < average_previous * 0.5:
        return (
            f"Rows dropped from ~{average_previous:.0f} to {latest}",
            IngestTrendStatus.warn,
        )

    return None, IngestTrendStatus.ok


def _compute_freshness_issue(
    values: Sequence[float | None],
) -> tuple[str | None, IngestTrendStatus]:
    if len(values) < 2:
        return None, IngestTrendStatus.ok

    latest = values[0]
    previous_values = [value for value in values[1:] if value is not None]
    if latest is None or not previous_values:
        return None, IngestTrendStatus.ok

    best_previous = min(previous_values)
    if best_previous <= 0:
        return None, IngestTrendStatus.ok

    if latest > best_previous * 1.5:
        return (
            f"Freshness regressed from {best_previous:.1f}s to {latest:.1f}s",
            IngestTrendStatus.warn,
        )

    return None, IngestTrendStatus.ok


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive guard
        return None
    if not text:
        return None
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _extract_rejected_records(metadata: Mapping[str, Any] | None) -> float | None:
    if not metadata:
        return None

    stack: list[Any] = [metadata]
    preferred: float | None = None
    fallback: float | None = None

    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                key_lower = str(key).lower()
                if "reject" in key_lower:
                    numeric = _coerce_float(value)
                    if numeric is not None:
                        safe_numeric = max(numeric, 0.0)
                        if "per_hour" in key_lower:
                            return safe_numeric
                        if any(token in key_lower for token in ("record", "records", "row", "rows", "count", "total")):
                            fallback = safe_numeric if fallback is None else fallback
                        elif preferred is None:
                            preferred = safe_numeric

                if isinstance(value, Mapping):
                    stack.append(value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    stack.extend(value)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            stack.extend(current)

    if fallback is not None:
        return fallback
    return preferred


def _record_dimension_metrics(
    dimension: str, records: Sequence[TimescaleIngestRunRecord]
) -> None:
    if not records:
        return

    latest = records[0]
    rejected = _extract_rejected_records(latest.metadata)
    if rejected is None:
        return

    duration_hours: float | None = None
    if latest.ingest_duration_seconds is not None and latest.ingest_duration_seconds > 0:
        duration_hours = float(latest.ingest_duration_seconds) / 3600.0

    if (duration_hours is None or duration_hours <= 0.0) and len(records) > 1:
        delta_seconds = (latest.executed_at - records[1].executed_at).total_seconds()
        if delta_seconds != 0:
            duration_hours = abs(delta_seconds) / 3600.0

    if duration_hours is None or duration_hours <= 0.0:
        duration_hours = 1.0

    rate = rejected / duration_hours if duration_hours > 0 else rejected
    operational_metrics.set_ingest_rejected_records_per_hour(dimension, rate)


def evaluate_ingest_trends(
    records: Iterable[TimescaleIngestRunRecord],
    *,
    lookback: int = 10,
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IngestTrendSnapshot:
    """Evaluate ingest trend posture across recent Timescale runs."""

    lookback = max(1, int(lookback))
    grouped: dict[str, list[TimescaleIngestRunRecord]] = {}
    for record in sorted(records, key=lambda item: item.executed_at, reverse=True):
        grouped.setdefault(record.dimension, []).append(record)

    generated_at = now or datetime.now(tz=UTC)
    if not grouped:
        return IngestTrendSnapshot(
            generated_at=generated_at,
            status=IngestTrendStatus.warn,
            lookback=lookback,
            dimensions=tuple(),
            issues=("No ingest history available",),
            metadata=dict(metadata or {}),
        )

    dimensions: list[IngestDimensionTrend] = []
    snapshot_issues: list[str] = []
    overall_status = IngestTrendStatus.ok

    for dimension, dimension_records in grouped.items():
        window = tuple(dimension_records[:lookback])
        statuses = tuple(_normalise_status(record.status) for record in window)
        rows = tuple(int(record.rows_written) for record in window)
        freshness = tuple(record.freshness_seconds for record in window)
        issues: list[str] = []
        status = statuses[0]

        if status is IngestTrendStatus.fail:
            issues.append("Latest run failed")
        elif status is IngestTrendStatus.warn:
            issues.append("Latest run completed with warnings")

        if len(statuses) > 1 and statuses[0] is statuses[1] is IngestTrendStatus.fail:
            status = IngestTrendStatus.fail
            issues.append("Two consecutive failures recorded")
        elif (
            any(s is IngestTrendStatus.fail for s in statuses[1:3])
            and status is not IngestTrendStatus.fail
        ):
            status = IngestTrendStatus.fail
            issues.append("Recent history includes failures")

        rows_issue, rows_severity = _compute_rows_issue(rows)
        if rows_issue:
            issues.append(rows_issue)
            status = _escalate(status, rows_severity)

        freshness_issue, freshness_severity = _compute_freshness_issue(freshness)
        if freshness_issue:
            issues.append(freshness_issue)
            status = _escalate(status, freshness_severity)

        metadata_payload = _build_dimension_metadata(window)
        if metadata:
            metadata_payload["context"] = dict(metadata)

        dimensions.append(
            IngestDimensionTrend(
                dimension=dimension,
                status=status,
                statuses=statuses,
                rows_written=rows,
                freshness_seconds=freshness,
                issues=tuple(issues),
                metadata=metadata_payload,
            )
        )

        _record_dimension_metrics(dimension, window)

        snapshot_issues.extend(f"{dimension}: {issue}" for issue in issues if issue)
        overall_status = _escalate(overall_status, status)

    return IngestTrendSnapshot(
        generated_at=generated_at,
        status=overall_status,
        lookback=lookback,
        dimensions=tuple(dimensions),
        issues=tuple(snapshot_issues),
        metadata=dict(metadata or {}),
    )


def publish_ingest_trends(
    event_bus: EventBus,
    snapshot: IngestTrendSnapshot,
    *,
    channel: str = "telemetry.ingest.trends",
) -> None:
    """Publish the ingest trend snapshot to event bus consumers."""

    event = Event(
        type=channel,
        payload=snapshot.as_dict(),
        source="ingest_trends",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message="Runtime event bus publish failed; falling back to global bus",
        runtime_unexpected_message=(
            "Unexpected error publishing ingest trend snapshot via runtime event bus"
        ),
        runtime_none_message=(
            "Runtime event bus returned no acknowledgement for ingest trend snapshot; "
            "falling back to global bus"
        ),
        global_not_running_message=(
            "Global event bus not running while publishing ingest trend snapshot"
        ),
        global_unexpected_message=(
            "Unexpected error publishing ingest trend snapshot via global event bus"
        ),
    )


__all__ = [
    "IngestDimensionTrend",
    "IngestTrendSnapshot",
    "IngestTrendStatus",
    "evaluate_ingest_trends",
    "format_ingest_trends_markdown",
    "publish_ingest_trends",
]
