"""Timescale data retention evaluation and telemetry helpers.

The roadmap's institutional data backbone stream calls out explicit data
retention targets so operators can prove the Timescale slice holds enough
history for professional tiers.【F:docs/EMP_ENCYCLOPEDIA_v2.3_CANONICAL.md†L6814-L6843】
This module grades the observed Timescale tables against those expectations and
publishes an aggregate snapshot that mirrors the rest of the backbone telemetry
surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.core.event_bus import Event, EventBus, get_global_bus


class RetentionStatus(Enum):
    """Severity levels exposed by data retention snapshots."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[RetentionStatus, int] = {
    RetentionStatus.ok: 0,
    RetentionStatus.warn: 1,
    RetentionStatus.fail: 2,
}


def _combine_status(current: RetentionStatus, candidate: RetentionStatus) -> RetentionStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class RetentionPolicy:
    """Declarative policy describing expected retention for a Timescale table."""

    dimension: str
    schema: str
    table: str
    target_days: int
    minimum_days: int
    optional: bool = False
    timestamp_column: str = "ts"
    description: str | None = None


@dataclass(frozen=True)
class RetentionComponentSnapshot:
    """Point-in-time view for a single retention check."""

    name: str
    status: RetentionStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class DataRetentionSnapshot:
    """Aggregated retention snapshot for institutional Timescale datasets."""

    status: RetentionStatus
    generated_at: datetime
    components: tuple[RetentionComponentSnapshot, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "components": [component.as_dict() for component in self.components],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.components:
            return "| Dataset | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Dataset | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            rows.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        return "\n".join(rows)


def _resolve_table_name(policy: RetentionPolicy, dialect: str) -> str:
    if dialect == "postgresql":
        return f"{policy.schema}.{policy.table}"
    return f"{policy.schema}_{policy.table}"


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _days_between(start: datetime, end: datetime) -> float:
    delta = end.astimezone(UTC) - start.astimezone(UTC)
    return max(delta.total_seconds() / 86_400.0, 0.0)


def evaluate_data_retention(
    engine: Engine,
    policies: Sequence[RetentionPolicy],
    *,
    reference: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataRetentionSnapshot:
    """Grade observed Timescale datasets against the configured retention policies."""

    reference_ts = reference or datetime.now(tz=UTC)
    components: list[RetentionComponentSnapshot] = []
    overall_status = RetentionStatus.ok

    with engine.begin() as conn:
        dialect = conn.dialect.name
        for policy in policies:
            table = _resolve_table_name(policy, dialect)
            stmt = text(
                f"SELECT MIN({policy.timestamp_column}) AS min_ts, "
                f"MAX({policy.timestamp_column}) AS max_ts, "
                f"COUNT(*) AS rowcount FROM {table}"
            )
            try:
                row = conn.execute(stmt).mappings().one()
            except Exception:
                components.append(
                    RetentionComponentSnapshot(
                        name=policy.dimension,
                        status=RetentionStatus.fail,
                        summary="Failed to query retention window",
                        metadata={"table": table},
                    )
                )
                overall_status = _combine_status(overall_status, RetentionStatus.fail)
                continue

            earliest = _parse_timestamp(row["min_ts"])
            latest = _parse_timestamp(row["max_ts"])
            rowcount = int(row["rowcount"] or 0)

            if rowcount <= 0 or earliest is None or latest is None:
                status = RetentionStatus.warn if policy.optional else RetentionStatus.fail
                summary = "No rows available"
                coverage_days = 0.0
                span_days = 0.0
            else:
                coverage_days = _days_between(earliest, reference_ts)
                span_days = _days_between(earliest, latest)
                if coverage_days >= policy.target_days:
                    status = RetentionStatus.ok
                    summary = f"Coverage {coverage_days:.1f}d (target {policy.target_days}d)"
                elif coverage_days >= policy.minimum_days:
                    status = RetentionStatus.warn
                    summary = f"Coverage {coverage_days:.1f}d below {policy.target_days}d target"
                else:
                    status = RetentionStatus.warn if policy.optional else RetentionStatus.fail
                    summary = f"Coverage {coverage_days:.1f}d below minimum {policy.minimum_days}d"

            metadata_payload: dict[str, object] = {
                "table": table,
                "rowcount": rowcount,
                "target_days": policy.target_days,
                "minimum_days": policy.minimum_days,
                "coverage_days": round(coverage_days, 2),
                "span_days": round(span_days, 2),
            }
            if earliest is not None:
                metadata_payload["earliest"] = earliest.astimezone(UTC).isoformat()
            if latest is not None:
                metadata_payload["latest"] = latest.astimezone(UTC).isoformat()
            if policy.optional:
                metadata_payload["optional"] = True
            if policy.description:
                metadata_payload["description"] = policy.description

            components.append(
                RetentionComponentSnapshot(
                    name=policy.dimension,
                    status=status,
                    summary=summary,
                    metadata=metadata_payload,
                )
            )
            overall_status = _combine_status(overall_status, status)

    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.setdefault(
        "policies",
        [
            {
                "dimension": policy.dimension,
                "target_days": policy.target_days,
                "minimum_days": policy.minimum_days,
                "optional": policy.optional,
            }
            for policy in policies
        ],
    )

    return DataRetentionSnapshot(
        status=overall_status,
        generated_at=reference_ts,
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_data_retention_markdown(snapshot: DataRetentionSnapshot) -> str:
    """Return a Markdown table summarising the retention snapshot."""

    return snapshot.to_markdown()


def publish_data_retention(
    event_bus: EventBus | None,
    snapshot: DataRetentionSnapshot,
) -> None:
    """Emit the retention snapshot on the runtime event bus."""

    event = Event(
        type="telemetry.data_backbone.retention",
        payload=snapshot.as_dict(),
        source="data_retention",
    )

    bus = event_bus or get_global_bus()
    publish_from_sync = getattr(bus, "publish_from_sync", None)
    is_running = getattr(bus, "is_running", None)
    if callable(publish_from_sync) and callable(is_running) and is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive publish fallback
            pass

    topic_bus = get_global_bus()
    topic_bus.publish_sync(event.type, event.payload, source=event.source)


__all__ = [
    "DataRetentionSnapshot",
    "RetentionComponentSnapshot",
    "RetentionPolicy",
    "RetentionStatus",
    "evaluate_data_retention",
    "format_data_retention_markdown",
    "publish_data_retention",
]
