import logging
from datetime import UTC, datetime, timedelta

from src.data_foundation.ingest.configuration import KafkaReadinessSettings
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaConsumerLagSnapshot,
    KafkaPartitionLag,
    KafkaTopicProvisioningSummary,
)
from src.operations.kafka_readiness import (
    KafkaReadinessStatus,
    evaluate_kafka_readiness,
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


def test_publish_kafka_readiness_logs_failures(monkeypatch, caplog) -> None:
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

        def publish_from_sync(self, event) -> None:
            raise RuntimeError("runtime bus failure")

    class _FailGlobalBus:
        def publish_sync(self, *_: object, **__: object) -> None:
            raise RuntimeError("global bus failure")

    monkeypatch.setattr(
        "src.operations.kafka_readiness.get_global_bus",
        lambda: _FailGlobalBus(),
    )

    with caplog.at_level(logging.WARNING):
        publish_kafka_readiness(_FailRuntimeBus(), snapshot)

    assert "Failed to publish Kafka readiness snapshot" in caplog.text
