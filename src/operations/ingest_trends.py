"""Ingest trend evaluation and telemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from statistics import mean
from typing import Iterable, Mapping, Sequence

from src.core.event_bus import Event, EventBus, get_global_bus
from src.data_foundation.persist.timescale import TimescaleIngestRunRecord


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

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            pass

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        pass


__all__ = [
    "IngestDimensionTrend",
    "IngestTrendSnapshot",
    "IngestTrendStatus",
    "evaluate_ingest_trends",
    "format_ingest_trends_markdown",
    "publish_ingest_trends",
]
