import logging
import pytest
from datetime import UTC, datetime, timedelta

from src.core.event_bus import Event
from src.data_foundation.ingest.configuration import KafkaReadinessSettings
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaConsumerLagSnapshot,
    KafkaPartitionLag,
    KafkaTopicProvisioningSummary,
)
from src.operations.event_bus_failover import EventPublishError
from src.operations.kafka_readiness import (
    KafkaReadinessStatus,
    evaluate_kafka_readiness,
    format_kafka_readiness_markdown,
    publish_kafka_readiness,
)


def _sample_connection(configured: bool = True) -> KafkaConnectionSettings:
    bootstrap = "localhost:9092" if configured else ""
    return KafkaConnectionSettings(bootstrap_servers=bootstrap)


def test_evaluate_kafka_readiness_ok() -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection()
    provisioning = KafkaTopicProvisioningSummary(
        requested=("ingest.daily",), existing=("ingest.daily",)
    )
    lag_snapshot = KafkaConsumerLagSnapshot(
        partitions=(
            KafkaPartitionLag(
                topic="ingest.daily",
                partition=0,
                current_offset=10,
                end_offset=20,
                lag=10,
            ),
        ),
        recorded_at=datetime.now(tz=UTC).isoformat(),
        total_lag=10,
        max_lag=10,
        topic_lag={"ingest.daily": 10},
    )

    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
        topics=("ingest.daily",),
        provisioning=provisioning,
        publishers=("events", "metrics"),
        lag_snapshot=lag_snapshot,
    )

    assert snapshot.status is KafkaReadinessStatus.ok
    assert snapshot.metadata["settings"]["enabled"] is True


def test_evaluate_kafka_readiness_flags_missing_brokers() -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection(configured=False)

    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
        topics=(),
        publishers=(),
    )

    assert snapshot.status is KafkaReadinessStatus.fail
    assert snapshot.components[0].status is KafkaReadinessStatus.fail


def test_evaluate_kafka_readiness_warns_on_stale_lag() -> None:
    settings = KafkaReadinessSettings(
        enabled=True,
        warn_stale_seconds=60.0,
        fail_stale_seconds=180.0,
    )
    connection = _sample_connection()
    recorded_at = datetime.now(tz=UTC) - timedelta(seconds=90)
    lag_snapshot = KafkaConsumerLagSnapshot(
        partitions=(
            KafkaPartitionLag(
                topic="ingest.intraday",
                partition=0,
                current_offset=100,
                end_offset=150,
                lag=50,
            ),
        ),
        recorded_at=recorded_at.isoformat(),
        total_lag=50,
        max_lag=50,
        topic_lag={"ingest.intraday": 50},
    )

    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
        topics=("ingest.intraday",),
        publishers=("events",),
        lag_snapshot=lag_snapshot,
    )

    assert snapshot.status is KafkaReadinessStatus.warn
    lag_component = next(
        component for component in snapshot.components if component.name == "consumer_lag"
    )
    assert "stale" in lag_component.summary


def test_evaluate_kafka_readiness_requires_topics_and_consumer() -> None:
    settings = KafkaReadinessSettings(
        enabled=True,
        min_publishers=0,
        require_topics=True,
        require_consumer=True,
    )
    connection = _sample_connection()

    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
        topics=(),
        publishers=(),
    )

    assert snapshot.status is KafkaReadinessStatus.fail
    topic_component = next(
        component for component in snapshot.components if component.name == "topics"
    )
    assert topic_component.status is KafkaReadinessStatus.fail
    consumer_component = next(
        component
        for component in snapshot.components
        if component.name == "consumer_lag"
    )
    assert consumer_component.status is KafkaReadinessStatus.fail
    assert consumer_component.metadata["required"] is True


def test_evaluate_kafka_readiness_handles_epoch_lag_timestamp() -> None:
    settings = KafkaReadinessSettings(
        enabled=True,
        warn_stale_seconds=60.0,
        fail_stale_seconds=120.0,
        min_publishers=0,
    )
    connection = _sample_connection()
    generated_at = datetime.now(tz=UTC)
    recorded_at = generated_at - timedelta(seconds=240)
    lag_snapshot = KafkaConsumerLagSnapshot(
        partitions=(
            KafkaPartitionLag(
                topic="ingest.daily",
                partition=0,
                current_offset=0,
                end_offset=0,
                lag=0,
            ),
        ),
        recorded_at=str(int(recorded_at.timestamp() * 1000)),
        total_lag=0,
        max_lag=0,
        topic_lag={"ingest.daily": 0},
    )

    snapshot = evaluate_kafka_readiness(
        generated_at=generated_at,
        settings=settings,
        connection=connection,
        topics=("ingest.daily",),
        publishers=(),
        lag_snapshot=lag_snapshot,
    )

    assert snapshot.status is KafkaReadinessStatus.fail
    lag_component = next(
        component for component in snapshot.components if component.name == "consumer_lag"
    )
    assert "stale" in lag_component.summary
    assert lag_component.metadata["lag_seconds"] > settings.fail_stale_seconds


def test_kafka_readiness_markdown_renders_components() -> None:
    settings = KafkaReadinessSettings(
        enabled=True,
        min_publishers=0,
        require_topics=False,
    )
    connection = _sample_connection()

    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
        topics=(" ingest.daily ", "ingest.daily", "analytics"),
        publishers=(),
    )

    topics_component = next(
        component for component in snapshot.components if component.name == "topics"
    )
    assert topics_component.metadata["expected"] == ["ingest.daily", "analytics"]

    markdown = format_kafka_readiness_markdown(snapshot)
    assert "| Component | Status | Summary |" in markdown
    assert "| topics | OK | Kafka topics configured |" in markdown


def test_publish_kafka_readiness_prefers_runtime_bus() -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection()
    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
    )

    class _StubEventBus:
        def __init__(self) -> None:
            self.events: list[Event] = []

        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> int:
            self.events.append(event)
            return 1

    bus = _StubEventBus()

    publish_kafka_readiness(bus, snapshot)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.kafka.readiness"
    assert event.payload["status"] == snapshot.status.value


def test_publish_kafka_readiness_falls_back_to_global_bus(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection()
    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
    )

    class _FailRuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> None:
            raise RuntimeError("runtime bus failure")

    class _TopicBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, object, str | None]] = []

        def publish_sync(
            self, topic: str, payload: object, *, source: str | None = None
        ) -> int:
            self.published.append((topic, payload, source))
            return 1

    topic_bus = _TopicBus()

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: topic_bus,
    )

    with caplog.at_level(logging.WARNING):
        publish_kafka_readiness(_FailRuntimeBus(), snapshot)

    assert "Runtime event bus unavailable for Kafka readiness" in caplog.text
    assert topic_bus.published


def test_publish_kafka_readiness_raises_on_unexpected_runtime_error() -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection()
    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
    )

    class _UnexpectedBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> int:
            raise ValueError("boom")

    with pytest.raises(EventPublishError) as excinfo:
        publish_kafka_readiness(_UnexpectedBus(), snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.kafka.readiness"


def test_publish_kafka_readiness_raises_when_global_bus_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = KafkaReadinessSettings(enabled=True)
    connection = _sample_connection()
    snapshot = evaluate_kafka_readiness(
        generated_at=datetime.now(tz=UTC),
        settings=settings,
        connection=connection,
    )

    class _RuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> int:
            raise RuntimeError("runtime failure")

    class _OfflineTopicBus:
        def publish_sync(self, *_: object, **__: object) -> int:
            raise RuntimeError("offline")

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: _OfflineTopicBus(),
    )

    with pytest.raises(EventPublishError) as excinfo:
        publish_kafka_readiness(_RuntimeBus(), snapshot)

    assert excinfo.value.stage == "global"
    assert excinfo.value.event_type == "telemetry.kafka.readiness"
