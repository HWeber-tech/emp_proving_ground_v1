"""Event bus health evaluation and telemetry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.core.event_bus import (
    Event,
    EventBus,
    EventBusStatistics,
    event_bus as global_event_bus,
    get_global_bus,
)
from src.operations.event_bus_failover import publish_event_with_failover


logger = logging.getLogger(__name__)


class EventBusHealthStatus(StrEnum):
    """Severity levels reported for event bus health."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[EventBusHealthStatus, int] = {
    EventBusHealthStatus.ok: 0,
    EventBusHealthStatus.warn: 1,
    EventBusHealthStatus.fail: 2,
}


def _escalate(
    current: EventBusHealthStatus, candidate: EventBusHealthStatus
) -> EventBusHealthStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class EventBusHealthSnapshot:
    """Aggregated event bus health telemetry."""

    service: str
    generated_at: datetime
    status: EventBusHealthStatus
    expected: bool
    running: bool
    loop_running: bool
    queue_size: int
    queue_capacity: int | None
    subscriber_count: int
    topic_subscribers: dict[str, int] = field(default_factory=dict)
    published_events: int = 0
    dropped_events: int = 0
    handler_errors: int = 0
    uptime_seconds: float | None = None
    last_event_at: datetime | None = None
    last_error_at: datetime | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "service": self.service,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "expected": self.expected,
            "running": self.running,
            "loop_running": self.loop_running,
            "queue_size": self.queue_size,
            "queue_capacity": self.queue_capacity,
            "subscriber_count": self.subscriber_count,
            "topic_subscribers": dict(self.topic_subscribers),
            "published_events": self.published_events,
            "dropped_events": self.dropped_events,
            "handler_errors": self.handler_errors,
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        if self.uptime_seconds is not None:
            payload["uptime_seconds"] = self.uptime_seconds
        if self.last_event_at is not None:
            payload["last_event_at"] = self.last_event_at.isoformat()
        if self.last_error_at is not None:
            payload["last_error_at"] = self.last_error_at.isoformat()
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"**Event bus health – {self.service}**",
            f"- Status: {self.status.value}",
            f"- Expected to run: {'yes' if self.expected else 'no'}",
            f"- Running: {'yes' if self.running else 'no'} (loop: {'yes' if self.loop_running else 'no'})",
            f"- Queue size: {self.queue_size}",
            f"- Queue capacity: {self.queue_capacity if self.queue_capacity is not None else 'unbounded'}",
            f"- Subscribers: {self.subscriber_count}",
            f"- Published events: {self.published_events}",
            f"- Dropped events: {self.dropped_events}",
            f"- Handler errors: {self.handler_errors}",
        ]
        if self.uptime_seconds is not None:
            lines.append(f"- Uptime seconds: {self.uptime_seconds:.0f}")
        if self.last_event_at is not None:
            lines.append(f"- Last event at: {self.last_event_at.isoformat()}")
        if self.last_error_at is not None:
            lines.append(f"- Last handler error at: {self.last_error_at.isoformat()}")
        if self.topic_subscribers:
            topics_summary = ", ".join(
                f"{topic}:{count}" for topic, count in sorted(self.topic_subscribers.items())
            )
            lines.append(f"- Topic subscribers: {topics_summary}")
        if self.metadata:
            for key, value in sorted(self.metadata.items()):
                lines.append(f"- {key}: {value}")
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def format_event_bus_markdown(snapshot: EventBusHealthSnapshot) -> str:
    """Convenience wrapper mirroring other telemetry formatters."""

    return snapshot.to_markdown()


def _convert_timestamp(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC)


def evaluate_event_bus_health(
    bus: EventBus | None = None,
    *,
    expected: bool = True,
    metadata: Mapping[str, object] | None = None,
    service: str = "event_bus",
    queue_warn_threshold: int = 100,
    queue_fail_threshold: int = 500,
    idle_warn_seconds: float = 300.0,
    now: datetime | None = None,
) -> EventBusHealthSnapshot:
    """Assess event bus health using observed statistics."""

    bus_instance = bus or global_event_bus
    stats: EventBusStatistics = bus_instance.get_statistics()
    moment = now or datetime.now(tz=UTC)

    running = stats.running and stats.loop_running
    issues: list[str] = []
    status = EventBusHealthStatus.ok

    if expected and not running:
        status = EventBusHealthStatus.fail
        issues.append("Event bus is not running; telemetry will be dropped")
    elif not running:
        status = EventBusHealthStatus.warn
        issues.append("Event bus is inactive")

    if stats.queue_size >= queue_fail_threshold:
        status = _escalate(status, EventBusHealthStatus.fail)
        issues.append(f"Queue has {stats.queue_size} pending events; worker may be stalled")
    elif stats.queue_size >= queue_warn_threshold:
        status = _escalate(status, EventBusHealthStatus.warn)
        issues.append(f"Queue has {stats.queue_size} pending events; monitor backlog")

    if stats.dropped_events > 0:
        level = EventBusHealthStatus.warn if stats.dropped_events < 5 else EventBusHealthStatus.fail
        status = _escalate(status, level)
        issues.append(f"{stats.dropped_events} events dropped while the bus was inactive")

    if stats.handler_errors > 0:
        level = EventBusHealthStatus.warn if stats.handler_errors < 3 else EventBusHealthStatus.fail
        status = _escalate(status, level)
        issues.append(
            f"{stats.handler_errors} handler error{'s' if stats.handler_errors != 1 else ''} recorded"
        )

    if expected and stats.published_events <= 0:
        status = _escalate(status, EventBusHealthStatus.warn)
        issues.append("No events published on the bus yet; confirm publishers are wired")

    last_event_at = _convert_timestamp(stats.last_event_timestamp)
    if (
        expected
        and stats.published_events > 0
        and last_event_at is not None
        and (moment - last_event_at).total_seconds() > idle_warn_seconds
    ):
        status = _escalate(status, EventBusHealthStatus.warn)
        issues.append("No events observed in the last five minutes")

    if expected and stats.subscriber_count <= 0:
        status = _escalate(status, EventBusHealthStatus.warn)
        issues.append("No subscribers registered; telemetry consumers may be missing")

    metadata_payload: dict[str, object] = {"expected": expected}
    if metadata:
        metadata_payload.update(dict(metadata))

    snapshot = EventBusHealthSnapshot(
        service=service,
        generated_at=moment,
        status=status,
        expected=expected,
        running=stats.running,
        loop_running=stats.loop_running,
        queue_size=stats.queue_size,
        queue_capacity=stats.queue_capacity,
        subscriber_count=stats.subscriber_count,
        topic_subscribers=dict(stats.topic_subscribers),
        published_events=stats.published_events,
        dropped_events=stats.dropped_events,
        handler_errors=stats.handler_errors,
        uptime_seconds=stats.uptime_seconds,
        last_event_at=last_event_at,
        last_error_at=_convert_timestamp(stats.last_error_timestamp),
        issues=tuple(issues),
        metadata=metadata_payload,
    )
    return snapshot


def publish_event_bus_health(event_bus: EventBus, snapshot: EventBusHealthSnapshot) -> None:
    """Publish the event bus health snapshot on the runtime event bus."""

    event = Event(
        type="telemetry.event_bus.health",
        payload=snapshot.as_dict(),
        source="operations.event_bus_health",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
        "Primary event bus publish_from_sync failed; falling back to global bus",
        runtime_unexpected_message=
        "Unexpected error publishing event bus health from primary bus",
        runtime_none_message=
        "Primary event bus publish_from_sync returned None; falling back to global bus",
        global_not_running_message=
        "Global event bus not running while publishing event bus health snapshot",
        global_unexpected_message=
        "Unexpected error publishing event bus health snapshot via global bus",
        global_bus_factory=get_global_bus,
    )


__all__ = [
    "EventBusHealthStatus",
    "EventBusHealthSnapshot",
    "evaluate_event_bus_health",
    "format_event_bus_markdown",
    "publish_event_bus_health",
]
