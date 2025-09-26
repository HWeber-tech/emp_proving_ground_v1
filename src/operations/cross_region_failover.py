"""Cross-region failover telemetry helpers aligned with the roadmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping

from src.core.event_bus import Event, EventBus
from src.data_foundation.ingest.configuration import TimescaleCrossRegionSettings
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    IngestSchedulerStatus,
)
from src.data_foundation.persist.timescale import (
    TimescaleIngestResult,
    TimescaleIngestRunRecord,
)
from src.operations.failover_drill import FailoverDrillSnapshot, FailoverDrillStatus


class CrossRegionStatus(StrEnum):
    """Severity levels used by the cross-region evaluation."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[CrossRegionStatus, int] = {
    CrossRegionStatus.ok: 0,
    CrossRegionStatus.warn: 1,
    CrossRegionStatus.fail: 2,
}


def _escalate(current: CrossRegionStatus, candidate: CrossRegionStatus) -> CrossRegionStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class CrossRegionComponent:
    """Individual component captured inside the cross-region snapshot."""

    name: str
    status: CrossRegionStatus
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
class CrossRegionFailoverSnapshot:
    """Aggregate snapshot describing cross-region replication posture."""

    status: CrossRegionStatus
    generated_at: datetime
    primary_region: str
    replica_region: str
    components: tuple[CrossRegionComponent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "primary_region": self.primary_region,
            "replica_region": self.replica_region,
            "components": [component.as_dict() for component in self.components],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        if not self.components:
            return "| Component | Status | Summary |\n| --- | --- | --- |\n"

        lines = ["| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            lines.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        return "\n".join(lines)


def _dimension_names(
    settings: TimescaleCrossRegionSettings,
    primary_results: Mapping[str, TimescaleIngestResult],
    replica_records: Mapping[str, TimescaleIngestRunRecord],
) -> tuple[str, ...]:
    if settings.dimensions:
        return settings.dimensions
    keys = list(primary_results.keys())
    for dimension in replica_records.keys():
        if dimension not in primary_results:
            keys.append(dimension)
    if not keys:
        return tuple()
    seen: set[str] = set()
    ordered: list[str] = []
    for dimension in keys:
        if dimension in seen:
            continue
        seen.add(dimension)
        ordered.append(dimension)
    return tuple(ordered)


def _evaluate_dimension(
    *,
    dimension: str,
    generated_at: datetime,
    primary: TimescaleIngestResult | None,
    replica: TimescaleIngestRunRecord | None,
    settings: TimescaleCrossRegionSettings,
) -> CrossRegionComponent:
    metadata: MutableMapping[str, object] = {
        "dimension": dimension,
        "warn_after_seconds": settings.warn_after_seconds,
        "fail_after_seconds": settings.fail_after_seconds,
        "max_row_difference_ratio": settings.max_row_difference_ratio,
    }

    if replica is None:
        return CrossRegionComponent(
            name=f"replica:{dimension}",
            status=CrossRegionStatus.fail,
            summary="Replica ingest record missing",
            metadata=metadata,
        )

    metadata.update(
        {
            "replica_status": replica.status,
            "replica_rows": replica.rows_written,
            "replica_executed_at": replica.executed_at.astimezone(UTC).isoformat(),
            "replica_symbols": list(replica.symbols),
        }
    )

    status = CrossRegionStatus.ok
    summary_parts: list[str] = []

    if replica.status not in {"ok", "skipped"}:
        status = CrossRegionStatus.warn
        summary_parts.append(f"status {replica.status}")

    delta = abs((generated_at - replica.executed_at).total_seconds())
    metadata["lag_seconds"] = delta
    if delta > settings.fail_after_seconds:
        status = CrossRegionStatus.fail
        summary_parts.append(f"lag {delta:.0f}s > fail threshold")
    elif delta > settings.warn_after_seconds:
        status = _escalate(status, CrossRegionStatus.warn)
        summary_parts.append(f"lag {delta:.0f}s > warn threshold")

    if primary is not None:
        metadata["primary_rows"] = primary.rows_written
        if primary.rows_written > 0:
            difference = abs(primary.rows_written - replica.rows_written)
            ratio = difference / max(primary.rows_written, 1)
            metadata["row_difference_ratio"] = ratio
            if ratio > settings.max_row_difference_ratio:
                status = _escalate(status, CrossRegionStatus.warn)
                summary_parts.append("row difference exceeds allowance")
        else:
            metadata["row_difference_ratio"] = 0.0

    if not summary_parts:
        summary_parts.append("replica healthy")

    return CrossRegionComponent(
        name=f"replica:{dimension}",
        status=status,
        summary="; ".join(summary_parts),
        metadata=metadata,
    )


def _failover_component(snapshot: FailoverDrillSnapshot | None) -> CrossRegionComponent | None:
    if snapshot is None:
        return None

    status_map = {
        FailoverDrillStatus.ok: CrossRegionStatus.ok,
        FailoverDrillStatus.warn: CrossRegionStatus.warn,
        FailoverDrillStatus.fail: CrossRegionStatus.fail,
    }
    status = status_map[snapshot.status]
    metadata = {
        "scenario": snapshot.scenario,
        "components": [component.as_dict() for component in snapshot.components],
    }
    summary = f"Failover drill status {snapshot.status.value}"
    return CrossRegionComponent(
        name="failover_drill",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def _scheduler_component(
    *,
    settings: TimescaleCrossRegionSettings,
    scheduler_snapshot: IngestSchedulerSnapshot | None,
    schedule_metadata: Mapping[str, object] | None,
) -> CrossRegionComponent:
    metadata: MutableMapping[str, object] = {
        "max_schedule_interval_seconds": settings.max_schedule_interval_seconds,
    }
    enabled = None
    interval = None
    if schedule_metadata:
        enabled = schedule_metadata.get("schedule_enabled")
        interval = schedule_metadata.get("schedule_interval_seconds")
        metadata.update(
            {
                "configured_enabled": enabled,
                "configured_interval_seconds": interval,
            }
        )

    if scheduler_snapshot is not None:
        metadata.update(
            {
                "snapshot": scheduler_snapshot.as_dict(),
            }
        )
        status_map = {
            IngestSchedulerStatus.ok: CrossRegionStatus.ok,
            IngestSchedulerStatus.warn: CrossRegionStatus.warn,
            IngestSchedulerStatus.fail: CrossRegionStatus.fail,
        }
        status = status_map[scheduler_snapshot.status]
        issues = list(scheduler_snapshot.issues)
        summary_parts = [f"Scheduler status {scheduler_snapshot.status.value}"]
        if issues:
            summary_parts.append("; ".join(issues))
        if (
            settings.max_schedule_interval_seconds is not None
            and scheduler_snapshot.interval_seconds > settings.max_schedule_interval_seconds
        ):
            status = _escalate(status, CrossRegionStatus.warn)
            summary_parts.append("interval exceeds cutover threshold")
        return CrossRegionComponent(
            name="scheduler",
            status=status,
            summary="; ".join(summary_parts),
            metadata=metadata,
        )

    summary = "Scheduler telemetry unavailable"
    status = CrossRegionStatus.warn
    if schedule_metadata and enabled is False:
        status = _escalate(status, CrossRegionStatus.warn)
        summary = "Scheduler disabled in configuration"
    return CrossRegionComponent(
        name="scheduler",
        status=status,
        summary=summary,
        metadata=metadata,
    )


def evaluate_cross_region_failover(
    *,
    generated_at: datetime,
    settings: TimescaleCrossRegionSettings,
    primary_results: Mapping[str, TimescaleIngestResult],
    replica_records: Mapping[str, TimescaleIngestRunRecord],
    scheduler_snapshot: IngestSchedulerSnapshot | None,
    schedule_metadata: Mapping[str, object] | None = None,
    failover_snapshot: FailoverDrillSnapshot | None = None,
    replica_error: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CrossRegionFailoverSnapshot:
    """Evaluate cross-region replication posture for the ingest slice."""

    components: list[CrossRegionComponent] = []

    if replica_error is not None:
        components.append(
            CrossRegionComponent(
                name="replica_connection",
                status=CrossRegionStatus.fail,
                summary="Replica ingest history unavailable",
                metadata={"error": replica_error},
            )
        )

    target_dimensions = _dimension_names(settings, primary_results, replica_records)
    for dimension in target_dimensions:
        component = _evaluate_dimension(
            dimension=dimension,
            generated_at=generated_at,
            primary=primary_results.get(dimension),
            replica=replica_records.get(dimension),
            settings=settings,
        )
        components.append(component)

    components.append(
        _scheduler_component(
            settings=settings,
            scheduler_snapshot=scheduler_snapshot,
            schedule_metadata=schedule_metadata,
        )
    )

    failover_component = _failover_component(failover_snapshot)
    if failover_component is not None:
        components.append(failover_component)

    status = CrossRegionStatus.ok
    for component in components:
        status = _escalate(status, component.status)

    snapshot_metadata: dict[str, object] = {
        "settings": settings.to_metadata(),
    }
    if metadata:
        snapshot_metadata.update(metadata)

    return CrossRegionFailoverSnapshot(
        status=status,
        generated_at=generated_at,
        primary_region=settings.primary_region,
        replica_region=settings.replica_region,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_cross_region_markdown(snapshot: CrossRegionFailoverSnapshot) -> str:
    """Return a Markdown representation of the snapshot."""

    return snapshot.to_markdown()


def publish_cross_region_snapshot(
    event_bus: EventBus, snapshot: CrossRegionFailoverSnapshot
) -> None:
    """Publish the snapshot on the ingest telemetry channel."""

    event_bus.publish_from_sync(
        Event(
            type="telemetry.ingest.cross_region_failover",
            payload=snapshot.as_dict(),
        )
    )
