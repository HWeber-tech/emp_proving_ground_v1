"""Kafka readiness telemetry helpers aligned with the modernization roadmap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.core.event_bus import Event, EventBus
from src.data_foundation.ingest.configuration import KafkaReadinessSettings
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaConsumerLagSnapshot,
    KafkaTopicProvisioningSummary,
)
from src.operations.event_bus_failover import publish_event_with_failover


logger = logging.getLogger(__name__)


class KafkaReadinessStatus(StrEnum):
    """Severity levels captured by the Kafka readiness snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[KafkaReadinessStatus, int] = {
    KafkaReadinessStatus.ok: 0,
    KafkaReadinessStatus.warn: 1,
    KafkaReadinessStatus.fail: 2,
}


def _escalate(
    current: KafkaReadinessStatus, candidate: KafkaReadinessStatus
) -> KafkaReadinessStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_topics(topics: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for topic in topics:
        key = str(topic).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return tuple(ordered)


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            seconds = float(text)
        except ValueError:
            return None
        if seconds > 1e12:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class KafkaReadinessComponent:
    """Individual readiness component captured inside the snapshot."""

    name: str
    status: KafkaReadinessStatus
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
class KafkaReadinessSnapshot:
    """Aggregate snapshot describing Kafka streaming readiness."""

    status: KafkaReadinessStatus
    generated_at: datetime
    brokers: str
    components: tuple[KafkaReadinessComponent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "brokers": self.brokers,
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


def evaluate_kafka_readiness(
    *,
    generated_at: datetime,
    settings: KafkaReadinessSettings,
    connection: KafkaConnectionSettings,
    topics: Sequence[str] = (),
    provisioning: KafkaTopicProvisioningSummary | None = None,
    publishers: Sequence[str] = (),
    lag_snapshot: KafkaConsumerLagSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
) -> KafkaReadinessSnapshot:
    """Evaluate Kafka readiness using connection info and telemetry inputs."""

    components: list[KafkaReadinessComponent] = []
    overall = KafkaReadinessStatus.ok

    connection_metadata = {
        "configured": connection.configured,
        "summary": connection.summary(redacted=True),
    }
    if connection.configured:
        status = KafkaReadinessStatus.ok
        summary = "Kafka brokers configured"
    else:
        status = KafkaReadinessStatus.fail
        summary = "Kafka brokers not configured"
    components.append(
        KafkaReadinessComponent(
            name="connection",
            status=status,
            summary=summary,
            metadata=connection_metadata,
        )
    )
    overall = _escalate(overall, status)

    ordered_topics = _normalise_topics(topics)
    topic_metadata: MutableMapping[str, object] = {
        "expected": list(ordered_topics),
        "required": settings.require_topics,
    }
    if provisioning is not None:
        topic_metadata["provisioning"] = provisioning.as_dict()
    if settings.require_topics and not ordered_topics:
        topic_status = KafkaReadinessStatus.fail
        topic_summary = "Kafka topics required but not configured"
    elif provisioning is not None and provisioning.failed:
        topic_status = KafkaReadinessStatus.fail
        topic_summary = "Kafka topic provisioning failures detected"
    elif ordered_topics:
        topic_status = KafkaReadinessStatus.ok
        topic_summary = "Kafka topics configured"
    else:
        topic_status = KafkaReadinessStatus.warn
        topic_summary = "Kafka topics optional for current run"
    components.append(
        KafkaReadinessComponent(
            name="topics",
            status=topic_status,
            summary=topic_summary,
            metadata=topic_metadata,
        )
    )
    overall = _escalate(overall, topic_status)

    publisher_metadata = {
        "active": list(publishers),
        "min_required": settings.min_publishers,
    }
    publisher_count = len(publishers)
    if publisher_count >= settings.min_publishers:
        publisher_status = KafkaReadinessStatus.ok
        publisher_summary = "Kafka publishers active"
    elif publisher_count == 0 and settings.min_publishers == 0:
        publisher_status = KafkaReadinessStatus.ok
        publisher_summary = "Kafka publishers optional"
    elif publisher_count == 0:
        publisher_status = KafkaReadinessStatus.fail
        publisher_summary = "Kafka publishers unavailable"
    else:
        publisher_status = KafkaReadinessStatus.warn
        publisher_summary = "Kafka publishers below target"
    components.append(
        KafkaReadinessComponent(
            name="publishers",
            status=publisher_status,
            summary=publisher_summary,
            metadata=publisher_metadata,
        )
    )
    overall = _escalate(overall, publisher_status)

    if lag_snapshot is not None:
        lag_metadata: MutableMapping[str, object] = {
            "recorded_at": lag_snapshot.recorded_at,
            "total_lag": lag_snapshot.total_lag,
            "max_lag": lag_snapshot.max_lag,
            "topics": dict(lag_snapshot.topic_lag),
        }
        lag_status = KafkaReadinessStatus.ok
        lag_summary_parts: list[str] = []

        if lag_snapshot.total_lag is not None:
            total = int(lag_snapshot.total_lag)
            lag_metadata["total_lag"] = total
            if total > settings.fail_lag_messages:
                lag_status = KafkaReadinessStatus.fail
                lag_summary_parts.append(
                    f"total lag {total} > fail threshold {settings.fail_lag_messages}"
                )
            elif total > settings.warn_lag_messages:
                lag_status = KafkaReadinessStatus.warn
                lag_summary_parts.append(
                    f"total lag {total} > warn threshold {settings.warn_lag_messages}"
                )

        recorded_at = _parse_timestamp(lag_snapshot.recorded_at)
        if recorded_at is not None:
            delta = abs((generated_at - recorded_at).total_seconds())
            lag_metadata["lag_seconds"] = delta
            if delta > settings.fail_stale_seconds:
                lag_status = KafkaReadinessStatus.fail
                lag_summary_parts.append(f"lag snapshot stale by {int(delta)}s > fail threshold")
            elif delta > settings.warn_stale_seconds:
                lag_status = _escalate(lag_status, KafkaReadinessStatus.warn)
                lag_summary_parts.append(f"lag snapshot stale by {int(delta)}s > warn threshold")
        else:
            lag_summary_parts.append("lag snapshot timestamp unavailable")

        if not lag_summary_parts:
            lag_summary_parts.append("lag within thresholds")

        components.append(
            KafkaReadinessComponent(
                name="consumer_lag",
                status=lag_status,
                summary="; ".join(lag_summary_parts),
                metadata=lag_metadata,
            )
        )
        overall = _escalate(overall, lag_status)
    elif settings.require_consumer:
        components.append(
            KafkaReadinessComponent(
                name="consumer_lag",
                status=KafkaReadinessStatus.fail,
                summary="Kafka consumer lag snapshot required but unavailable",
                metadata={"required": True},
            )
        )
        overall = KafkaReadinessStatus.fail

    snapshot_metadata: MutableMapping[str, object] = {
        "settings": settings.to_metadata(),
    }
    if metadata:
        snapshot_metadata.update(dict(metadata))

    return KafkaReadinessSnapshot(
        status=overall,
        generated_at=generated_at.astimezone(UTC),
        brokers=connection.summary(redacted=True),
        components=tuple(components),
        metadata=snapshot_metadata,
    )


def format_kafka_readiness_markdown(snapshot: KafkaReadinessSnapshot) -> str:
    """Return a Markdown rendering of the Kafka readiness snapshot."""

    return snapshot.to_markdown()


def publish_kafka_readiness(event_bus: EventBus, snapshot: KafkaReadinessSnapshot) -> None:
    """Publish the Kafka readiness snapshot on the event bus."""

    event = Event(
        type="telemetry.kafka.readiness",
        payload=snapshot.as_dict(),
        source="operations.kafka_readiness",
    )

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=
            "Runtime event bus unavailable for Kafka readiness; falling back to global bus",
        runtime_unexpected_message=
            "Unexpected error publishing Kafka readiness via runtime event bus",
        runtime_none_message=
            "Runtime event bus returned no result for Kafka readiness; falling back to global bus",
        global_not_running_message=
            "Global event bus unavailable while publishing Kafka readiness snapshot",
        global_unexpected_message=
            "Unexpected error publishing Kafka readiness snapshot via global bus",
    )
