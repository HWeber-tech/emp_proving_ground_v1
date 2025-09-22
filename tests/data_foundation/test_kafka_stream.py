import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.quality import evaluate_ingest_quality
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
    TimescaleIngestor,
    TimescaleMigrator,
)
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaConsumerLagSnapshot,
    KafkaIngestBackfillSummary,
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
    KafkaIngestHealthPublisher,
    KafkaIngestMetricsPublisher,
    KafkaIngestQualityPublisher,
    KafkaPartitionLag,
    KafkaTopicProvisioner,
    KafkaTopicSpec,
    backfill_ingest_dimension_to_kafka,
    capture_consumer_lag,
    create_ingest_event_consumer,
    create_ingest_event_publisher,
    create_ingest_health_publisher,
    create_ingest_metrics_publisher,
    create_ingest_quality_publisher,
    ingest_topic_config_from_mapping,
    resolve_ingest_topic_specs,
    should_auto_create_topics,
)
from src.data_foundation.ingest.metrics import summarise_ingest_metrics


class _FakeProducer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bytes, str | bytes | None]] = []
        self.flush_calls: list[float | None] = []

    def produce(self, topic: str, value: bytes, key: str | bytes | None = None) -> None:
        self.messages.append((topic, value, key))

    def flush(self, timeout: float | None = None) -> None:
        self.flush_calls.append(timeout)


class _FakeKafkaMessage:
    def __init__(
        self,
        value: bytes | str | None,
        *,
        topic: str = "timescale.daily",
        key: bytes | str | None = None,
        error: object | None = None,
    ) -> None:
        self._value = value
        self._topic = topic
        self._key = key
        self._error = error

    def error(self) -> object | None:
        return self._error

    def value(self) -> bytes | str | None:
        return self._value

    def topic(self) -> str:
        return self._topic

    def key(self) -> bytes | str | None:
        return self._key


class _FakeConsumer:
    def __init__(self) -> None:
        self.subscriptions: list[list[str]] = []
        self.messages: list[_FakeKafkaMessage] = []
        self.closed = False
        self.commits: list[dict[str, object]] = []
        self.metrics_data: Mapping[str, object] | None = None
        self.metrics_calls = 0

    def subscribe(self, topics: Mapping[str, object] | list[str]) -> None:  # type: ignore[override]
        self.subscriptions.append(list(topics))

    def poll(self, timeout: float | None = None) -> _FakeKafkaMessage | None:
        if self.messages:
            return self.messages.pop(0)
        return None

    def close(self) -> None:
        self.closed = True

    def commit(
        self,
        message: _FakeKafkaMessage | None = None,
        asynchronous: bool = False,
    ) -> None:
        self.commits.append(
            {
                "message": message,
                "asynchronous": asynchronous,
            }
        )

    def metrics(self) -> Mapping[str, object]:
        self.metrics_calls += 1
        return self.metrics_data or {}


class _ImmediateFuture:
    def result(self, timeout: float | None = None) -> None:  # pragma: no cover - trivial
        return None


class _FakeAdminClient:
    def __init__(self, *, existing: Sequence[str] | None = None) -> None:
        self.topics = {name: object() for name in (existing or [])}
        self.created: list[KafkaTopicSpec] = []
        self.list_calls = 0
        self.create_calls = 0

    def list_topics(self, timeout: float | None = None) -> object:
        self.list_calls += 1

        class _Metadata:
            def __init__(self, topics: Mapping[str, object]) -> None:
                self.topics = topics

        return _Metadata(self.topics)

    def create_topics(
        self,
        new_topics: Sequence[KafkaTopicSpec],
        request_timeout: float | None = None,
    ) -> Mapping[str, _ImmediateFuture]:
        self.create_calls += 1
        futures: dict[str, _ImmediateFuture] = {}
        for spec in new_topics:
            self.created.append(spec)
            self.topics[spec.name] = object()
            futures[spec.name] = _ImmediateFuture()
        return futures


def _build_consumer_metrics() -> Mapping[str, object]:
    return {
        "time": 1_700_000_000,
        "client_id": "emp-ingest-bridge",
        "group_id": "emp-ingest-group",
        "topics": {
            "timescale.daily": {
                "partitions": {
                    "0": {
                        "consumer_lag": 5,
                        "consumer_offset": 10,
                        "hi_offset": 15,
                    },
                    "1": {
                        "lag": 2,
                        "current_offset": 3,
                        "end_offset": 5,
                    },
                }
            }
        },
    }


@pytest.fixture()
def fake_result() -> TimescaleIngestResult:
    return TimescaleIngestResult(
        rows_written=5,
        symbols=("EURUSD", "GBPUSD"),
        start_ts=datetime(2024, 1, 2, tzinfo=UTC),
        end_ts=datetime(2024, 1, 3, tzinfo=UTC),
        ingest_duration_seconds=0.45,
        freshness_seconds=180.0,
        dimension="daily_bars",
        source="yahoo",
    )


@pytest.fixture()
def sample_health_report() -> IngestHealthReport:
    generated = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    check = IngestHealthCheck(
        dimension="daily_bars",
        status=IngestHealthStatus.ok,
        message="Ingest healthy",
        rows_written=5,
        freshness_seconds=180.0,
        expected_symbols=("EURUSD",),
        observed_symbols=("EURUSD",),
        missing_symbols=tuple(),
        ingest_duration_seconds=0.45,
        metadata={"freshness_sla_seconds": 86_400.0},
    )
    return IngestHealthReport(
        status=IngestHealthStatus.ok,
        generated_at=generated,
        checks=(check,),
        metadata={
            "planned_dimensions": ["daily_bars"],
            "observed_dimensions": ["daily_bars"],
        },
    )


def test_kafka_connection_settings_from_mapping() -> None:
    settings = KafkaConnectionSettings.from_mapping(
        {
            "KAFKA_BROKERS": "broker1:9092, broker2:9092 ",
            "KAFKA_USERNAME": "emp",
            "KAFKA_PASSWORD": "secret",
            "KAFKA_CLIENT_ID": "emp-ingest",
            "KAFKA_LINGER_MS": "15",
            "KAFKA_BATCH_SIZE": "65536",
            "KAFKA_COMPRESSION_TYPE": "snappy",
        }
    )

    assert settings.bootstrap_servers == "broker1:9092,broker2:9092"
    assert settings.security_protocol == "SASL_SSL"
    assert settings.sasl_mechanism == "PLAIN"
    assert settings.username == "emp"
    assert settings.password == "secret"
    assert settings.client_id == "emp-ingest"
    assert settings.linger_ms == 15
    assert settings.batch_size == 65536
    assert settings.compression_type == "snappy"

    config = settings.client_config()
    assert config["bootstrap.servers"] == "broker1:9092,broker2:9092"
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanism"] == "PLAIN"
    assert config["sasl.username"] == "emp"
    assert config["sasl.password"] == "secret"
    assert config["compression.type"] == "snappy"
    assert config["enable.idempotence"] is True

    consumer_config = settings.consumer_config(
        group_id="bridge",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    assert consumer_config["group.id"] == "bridge"
    assert consumer_config["auto.offset.reset"] == "earliest"
    assert consumer_config["enable.auto.commit"] is False


def test_kafka_connection_settings_create_producer() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    created: dict[str, object] = {}

    def factory(config: dict[str, object]) -> str:
        created["config"] = config
        return "producer"

    producer = settings.create_producer(factory=factory)
    assert producer == "producer"
    assert created["config"]["bootstrap.servers"] == "localhost:9092"


def test_kafka_connection_settings_create_producer_unconfigured(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({})

    with caplog.at_level("DEBUG"):
        producer = settings.create_producer(factory=lambda config: "unused")

    assert producer is None
    assert any("Kafka connection not configured" in message for message in caplog.messages)


def test_kafka_ingest_event_publisher_emits(fake_result: TimescaleIngestResult) -> None:
    producer = _FakeProducer()
    publisher = KafkaIngestEventPublisher(
        producer,
        topic_map={"daily_bars": "timescale.daily", "intraday_trades": "timescale.intraday"},
        default_topic="timescale.default",
        flush_timeout=1.5,
    )

    publisher.publish(fake_result, metadata={"plan": "daily_bars", "lookback_days": 30})

    assert len(producer.messages) == 1
    topic, value, key = producer.messages[0]
    assert topic == "timescale.daily"
    assert key == "EURUSD"

    payload = json.loads(value.decode("utf-8"))
    assert payload["result"]["dimension"] == "daily_bars"
    assert payload["result"]["source"] == "yahoo"
    assert payload["metadata"]["plan"] == "daily_bars"
    assert payload["metadata"]["lookback_days"] == 30
    assert "emitted_at" in payload

    assert producer.flush_calls == [1.5]


def test_kafka_ingest_event_publisher_skips_without_topic(
    fake_result: TimescaleIngestResult,
) -> None:
    producer = _FakeProducer()
    publisher = KafkaIngestEventPublisher(producer, topic_map={})

    publisher.publish(fake_result, metadata={"plan": "daily_bars"})

    assert producer.messages == []
    assert producer.flush_calls == []


def test_kafka_ingest_health_publisher_emits(sample_health_report: IngestHealthReport) -> None:
    producer = _FakeProducer()
    publisher = KafkaIngestHealthPublisher(
        producer,
        topic="telemetry.ingest.health",
        key="health",
        flush_timeout=0.75,
    )

    publisher.publish(
        sample_health_report,
        metadata={"runtime": "professional", "generated": datetime(2024, 1, 3, tzinfo=UTC)},
    )

    assert len(producer.messages) == 1
    topic, value, key = producer.messages[0]
    assert topic == "telemetry.ingest.health"
    assert key == "health"

    payload = json.loads(value.decode("utf-8"))
    assert payload["report"]["status"] == "ok"
    assert payload["report"]["checks"][0]["dimension"] == "daily_bars"
    assert payload["metadata"]["runtime"] == "professional"
    assert payload["metadata"]["generated"].endswith("+00:00")
    assert producer.flush_calls == [0.75]
    assert "telemetry.ingest.health" in publisher.summary()


def test_ingest_topic_config_from_mapping_parses_sources() -> None:
    mapping = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily, intraday_trades=timescale.intraday",
        "KAFKA_INGEST_TOPIC_MACRO_EVENTS": "timescale.macro",
        "KAFKA_INGEST_DEFAULT_TOPIC": "timescale.default",
    }

    topic_map, default_topic = ingest_topic_config_from_mapping(mapping)

    assert topic_map == {
        "daily_bars": "timescale.daily",
        "intraday_trades": "timescale.intraday",
        "macro_events": "timescale.macro",
    }
    assert default_topic == "timescale.default"


def test_kafka_ingest_event_consumer_publishes_to_bus() -> None:
    class _RecordingBus:
        def __init__(self) -> None:
            self.events: list = []

        def publish_from_sync(self, event) -> int:
            self.events.append(event)
            return 1

        def is_running(self) -> bool:
            return True

    bus = _RecordingBus()

    consumer = _FakeConsumer()
    payload = {
        "result": {"dimension": "daily_bars", "rows_written": 5},
        "metadata": {"plan": "daily_bars"},
    }
    consumer.messages.append(
        _FakeKafkaMessage(json.dumps(payload), topic="timescale.daily", key=b"EURUSD")
    )

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.01,
        idle_sleep=0.0,
    )

    processed = bridge.poll_once()
    assert processed is True
    assert consumer.closed is False
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.ingest"
    assert event.payload["result"]["dimension"] == "daily_bars"
    assert event.payload["metadata"]["plan"] == "daily_bars"
    assert event.payload["kafka_topic"] == "timescale.daily"
    assert event.payload["kafka_key"] == "EURUSD"
    assert "kafka_received_at" in event.payload


def test_kafka_ingest_event_consumer_commits_offsets_when_enabled() -> None:
    class _RecordingBus:
        def __init__(self) -> None:
            self.events: list = []

        def publish_from_sync(self, event) -> int:
            self.events.append(event)
            return 1

        def is_running(self) -> bool:
            return True

    bus = _RecordingBus()
    consumer = _FakeConsumer()
    message = _FakeKafkaMessage(json.dumps({"result": {"dimension": "daily_bars"}}))
    consumer.messages.append(message)

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.01,
        idle_sleep=0.0,
        commit_offsets=True,
        commit_asynchronously=False,
    )

    processed = bridge.poll_once()
    assert processed is True
    assert len(bus.events) == 1
    assert len(consumer.commits) == 1
    commit_record = consumer.commits[0]
    assert commit_record["message"] is None or commit_record["message"] == message
    assert commit_record["asynchronous"] is False


def test_capture_consumer_lag_from_metrics() -> None:
    consumer = _FakeConsumer()
    consumer.metrics_data = _build_consumer_metrics()

    snapshot = capture_consumer_lag(consumer)

    assert snapshot is not None
    assert isinstance(snapshot, KafkaConsumerLagSnapshot)
    assert snapshot.total_lag == 7
    assert snapshot.max_lag == 5
    assert snapshot.topic_lag == {"timescale.daily": 7}
    assert snapshot.metadata["client_id"] == "emp-ingest-bridge"

    partitions = list(snapshot.partitions)
    assert partitions[0] == KafkaPartitionLag(
        topic="timescale.daily",
        partition=0,
        current_offset=10,
        end_offset=15,
        lag=5,
    )
    payload = snapshot.as_dict()
    assert payload["total_lag"] == 7
    assert payload["partitions"][0]["lag"] == 5


def test_kafka_ingest_event_consumer_publishes_consumer_lag() -> None:
    class _RecordingBus:
        def __init__(self) -> None:
            self.events: list = []

        def publish_from_sync(self, event) -> int:
            self.events.append(event)
            return 1

        def is_running(self) -> bool:
            return True

    bus = _RecordingBus()

    consumer = _FakeConsumer()
    consumer.metrics_data = _build_consumer_metrics()
    payload = {"result": {"dimension": "daily_bars"}}
    consumer.messages.append(_FakeKafkaMessage(json.dumps(payload)))

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.01,
        idle_sleep=0.0,
        publish_consumer_lag=True,
        consumer_lag_event_type="telemetry.kafka.lag",
        consumer_lag_source="timescale_ingest.kafka.lag",
        consumer_lag_interval=0.0,
    )

    processed = bridge.poll_once()

    assert processed is True
    assert consumer.metrics_calls == 1
    assert len(bus.events) == 2
    lag_event = bus.events[1]
    assert lag_event.type == "telemetry.kafka.lag"
    assert lag_event.payload["total_lag"] == 7
    assert lag_event.payload["topics"] == ["timescale.daily"]


def test_kafka_ingest_event_consumer_respects_consumer_lag_interval(monkeypatch) -> None:
    class _RecordingBus:
        def __init__(self) -> None:
            self.events: list = []

        def publish_from_sync(self, event) -> int:
            self.events.append(event)
            return 1

        def is_running(self) -> bool:
            return True

    bus = _RecordingBus()
    consumer = _FakeConsumer()
    consumer.metrics_data = _build_consumer_metrics()
    consumer.messages.append(_FakeKafkaMessage(json.dumps({"result": {"dimension": "daily"}})))
    consumer.messages.append(_FakeKafkaMessage(json.dumps({"result": {"dimension": "daily"}})))

    times = iter([1.0, 1.4, 120.0])

    def _fake_monotonic() -> float:
        try:
            return next(times)
        except StopIteration:
            return 120.0

    monkeypatch.setattr(
        "src.data_foundation.streaming.kafka_stream.time.monotonic", _fake_monotonic
    )

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.01,
        idle_sleep=0.0,
        publish_consumer_lag=True,
        consumer_lag_interval=60.0,
    )

    bridge.poll_once()
    bridge.poll_once()

    event_types = [event.type for event in bus.events]
    assert event_types.count("telemetry.kafka.lag") == 1
    assert consumer.metrics_calls == 1


def test_kafka_ingest_event_consumer_handles_errors(caplog) -> None:
    bus = EventBus()
    consumer = _FakeConsumer()
    consumer.messages.append(_FakeKafkaMessage(None))
    consumer.messages.append(_FakeKafkaMessage(b"not-json"))
    consumer.messages.append(_FakeKafkaMessage(b"{}", error="failure"))

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.0,
        idle_sleep=0.0,
    )

    with caplog.at_level("WARNING"):
        bridge.poll_once()
        bridge.poll_once()
        bridge.poll_once()

    assert any("decode" in message for message in caplog.messages)
    assert any("error" in message for message in caplog.messages)


@pytest.mark.asyncio()
async def test_kafka_ingest_event_consumer_run_forever_stops() -> None:
    bus = EventBus()
    consumer = _FakeConsumer()
    consumer.messages.append(_FakeKafkaMessage(json.dumps({"result": {"dimension": "daily_bars"}})))

    bridge = KafkaIngestEventConsumer(
        consumer,
        topics=["timescale.daily"],
        event_bus=bus,
        poll_timeout=0.01,
        idle_sleep=0.0,
    )

    stop_event = asyncio.Event()

    async def _trigger_stop() -> None:
        await asyncio.sleep(0.05)
        stop_event.set()

    task = asyncio.create_task(bridge.run_forever(stop_event))
    await _trigger_stop()
    await task
    assert consumer.closed is True


def test_create_ingest_event_publisher_configures(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_FLUSH_TIMEOUT": "off",
    }

    created: dict[str, object] = {}

    def factory(config: Mapping[str, object]) -> _FakeProducer:
        created["config"] = dict(config)
        return _FakeProducer()

    with caplog.at_level("DEBUG"):
        publisher = create_ingest_event_publisher(settings, extras, producer_factory=factory)

    assert publisher is not None
    assert isinstance(publisher, KafkaIngestEventPublisher)
    assert created["config"]["bootstrap.servers"] == "localhost:9092"


def test_create_ingest_event_publisher_warns_without_topics(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with caplog.at_level("WARNING"):
        publisher = create_ingest_event_publisher(
            settings,
            {},
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("no ingest topics" in message for message in caplog.messages)


def test_create_ingest_event_consumer_configures() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_CONSUMER_GROUP": "ingest-bridge",
    }

    created: dict[str, Mapping[str, object]] = {}

    def factory(config: Mapping[str, object]) -> _FakeConsumer:
        created["config"] = dict(config)
        return _FakeConsumer()

    consumer = create_ingest_event_consumer(
        settings,
        extras,
        event_bus=EventBus(),
        consumer_factory=factory,
    )

    assert consumer is not None
    assert isinstance(consumer, KafkaIngestEventConsumer)
    assert created["config"]["group.id"] == "ingest-bridge"
    assert consumer.topics == ("timescale.daily",)


def test_create_ingest_event_consumer_disabled(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_CONSUMER_ENABLED": "false",
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
    }

    with caplog.at_level("INFO"):
        consumer = create_ingest_event_consumer(
            settings,
            extras,
            event_bus=EventBus(),
            consumer_factory=lambda config: _FakeConsumer(),
        )

    assert consumer is None
    assert any("disabled" in message for message in caplog.messages)


def test_create_ingest_event_consumer_requires_topics(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with caplog.at_level("WARNING"):
        consumer = create_ingest_event_consumer(
            settings,
            {},
            event_bus=EventBus(),
            consumer_factory=lambda config: _FakeConsumer(),
        )

    assert consumer is None
    assert any("no topics" in message.lower() for message in caplog.messages)


def test_create_ingest_event_consumer_commits_when_auto_commit_disabled() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_CONSUMER_AUTO_COMMIT": "false",
    }

    created: dict[str, Mapping[str, object]] = {}
    consumer = _FakeConsumer()

    def factory(config: Mapping[str, object]) -> _FakeConsumer:
        created["config"] = dict(config)
        return consumer

    class _RecordingBus:
        def publish_from_sync(self, event) -> int:
            return 1

        def is_running(self) -> bool:
            return True

    bridge = create_ingest_event_consumer(
        settings,
        extras,
        event_bus=_RecordingBus(),
        consumer_factory=factory,
    )

    assert bridge is not None
    consumer.messages.append(
        _FakeKafkaMessage(json.dumps({"result": {"dimension": "intraday_trades"}}))
    )
    processed = bridge.poll_once()
    assert processed is True
    assert len(consumer.commits) == 1
    assert created["config"].get("enable.auto.commit") is False


def test_create_ingest_event_consumer_respects_commit_toggle() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_CONSUMER_AUTO_COMMIT": "false",
        "KAFKA_INGEST_CONSUMER_COMMIT_ON_PUBLISH": "no",
    }

    consumer = _FakeConsumer()

    def factory(config: Mapping[str, object]) -> _FakeConsumer:
        return consumer

    class _RecordingBus:
        def publish_from_sync(self, event) -> int:
            return 1

        def is_running(self) -> bool:
            return True

    bridge = create_ingest_event_consumer(
        settings,
        extras,
        event_bus=_RecordingBus(),
        consumer_factory=factory,
    )

    assert bridge is not None
    consumer.messages.append(_FakeKafkaMessage(json.dumps({"result": {"dimension": "daily_bars"}})))
    processed = bridge.poll_once()
    assert processed is True
    assert consumer.commits == []


def test_create_ingest_event_consumer_configures_async_commit() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_CONSUMER_AUTO_COMMIT": "false",
        "KAFKA_INGEST_CONSUMER_COMMIT_ASYNC": "true",
    }

    consumer = _FakeConsumer()

    def factory(config: Mapping[str, object]) -> _FakeConsumer:
        return consumer

    class _RecordingBus:
        def publish_from_sync(self, event) -> int:
            return 1

        def is_running(self) -> bool:
            return True

    bridge = create_ingest_event_consumer(
        settings,
        extras,
        event_bus=_RecordingBus(),
        consumer_factory=factory,
    )

    assert bridge is not None
    consumer.messages.append(_FakeKafkaMessage(json.dumps({"result": {"dimension": "daily_bars"}})))
    processed = bridge.poll_once()
    assert processed is True
    assert len(consumer.commits) == 1
    assert consumer.commits[0]["asynchronous"] is True


def test_create_ingest_health_publisher_configures() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_HEALTH_TOPIC": "telemetry.ingest.health",
        "KAFKA_INGEST_HEALTH_KEY": "health",
        "KAFKA_INGEST_HEALTH_FLUSH_TIMEOUT": "0.5",
    }

    created: dict[str, object] = {}

    def factory(config: Mapping[str, object]) -> _FakeProducer:
        created["config"] = dict(config)
        return _FakeProducer()

    publisher = create_ingest_health_publisher(settings, extras, producer_factory=factory)

    assert publisher is not None
    assert isinstance(publisher, KafkaIngestHealthPublisher)
    assert publisher.topic == "telemetry.ingest.health"
    assert publisher.key == "health"
    assert created["config"]["bootstrap.servers"] == "localhost:9092"


def test_create_ingest_health_publisher_disabled(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_HEALTH_TOPIC": "telemetry.ingest.health",
        "KAFKA_INGEST_HEALTH_ENABLED": "false",
    }

    with caplog.at_level("INFO"):
        publisher = create_ingest_health_publisher(
            settings,
            extras,
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("disabled" in message for message in caplog.messages)


def test_create_ingest_health_publisher_requires_topic(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with caplog.at_level("WARNING"):
        publisher = create_ingest_health_publisher(
            settings,
            {},
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("health publisher skipped" in message.lower() for message in caplog.messages)


def test_kafka_ingest_metrics_publisher_emits_payload(fake_result) -> None:
    producer = _FakeProducer()
    publisher = KafkaIngestMetricsPublisher(
        producer,
        topic="telemetry.ingest.metrics",
        key="metrics",
        flush_timeout=0.1,
    )
    snapshot = summarise_ingest_metrics({"daily_bars": fake_result})

    publisher.publish(snapshot, metadata={"plan": {"symbols": ("EURUSD",)}})

    assert producer.messages
    topic, payload, key = producer.messages[0]
    assert topic == "telemetry.ingest.metrics"
    assert key == "metrics"
    decoded = json.loads(payload.decode("utf-8"))
    assert decoded["metrics"]["total_rows"] == snapshot.total_rows()
    assert decoded["metrics"]["dimensions"][0]["dimension"] == "daily_bars"
    assert decoded["metadata"]["plan"]["symbols"] == ["EURUSD"]
    assert producer.flush_calls == [0.1]


def test_kafka_ingest_quality_publisher_emits_payload(fake_result) -> None:
    producer = _FakeProducer()
    publisher = KafkaIngestQualityPublisher(
        producer,
        topic="telemetry.ingest.quality",
        key="quality",
        flush_timeout=0.25,
    )
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"], lookback_days=1)
    )
    report = evaluate_ingest_quality({"daily_bars": fake_result}, plan=plan)

    publisher.publish(report, metadata={"plan": {"symbols": ("EURUSD",)}})

    assert producer.messages
    topic, payload, key = producer.messages[0]
    assert topic == "telemetry.ingest.quality"
    assert key == "quality"
    decoded = json.loads(payload.decode("utf-8"))
    assert decoded["quality"]["status"] == report.status.value
    assert decoded["metadata"]["plan"]["symbols"] == ["EURUSD"]
    assert producer.flush_calls == [0.25]


def test_create_ingest_metrics_publisher_configured() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_METRICS_TOPIC": "telemetry.ingest.metrics",
        "KAFKA_INGEST_METRICS_KEY": "metrics",
        "KAFKA_INGEST_METRICS_FLUSH_TIMEOUT": "off",
    }

    created: dict[str, Mapping[str, object]] = {}

    def factory(config: Mapping[str, object]) -> _FakeProducer:
        created["config"] = config
        return _FakeProducer()

    publisher = create_ingest_metrics_publisher(
        settings,
        extras,
        producer_factory=factory,
    )

    assert publisher is not None
    assert isinstance(publisher, KafkaIngestMetricsPublisher)
    assert publisher.topic == "telemetry.ingest.metrics"
    assert publisher.key == "metrics"
    assert publisher.flush_timeout is None
    assert created["config"]["bootstrap.servers"] == "localhost:9092"


def test_create_ingest_metrics_publisher_disabled(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_METRICS_TOPIC": "telemetry.ingest.metrics",
        "KAFKA_INGEST_METRICS_ENABLED": "0",
    }

    with caplog.at_level("INFO"):
        publisher = create_ingest_metrics_publisher(
            settings,
            extras,
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("metrics publisher disabled" in message for message in caplog.messages)


def test_create_ingest_metrics_publisher_requires_topic(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with caplog.at_level("WARNING"):
        publisher = create_ingest_metrics_publisher(
            settings,
            {},
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("metrics publisher skipped" in message.lower() for message in caplog.messages)


def test_create_ingest_quality_publisher_configured() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_QUALITY_TOPIC": "telemetry.ingest.quality",
        "KAFKA_INGEST_QUALITY_KEY": "quality",
        "KAFKA_INGEST_QUALITY_FLUSH_TIMEOUT": "off",
    }

    created: dict[str, Mapping[str, object]] = {}

    def factory(config: Mapping[str, object]) -> _FakeProducer:
        created["config"] = config
        return _FakeProducer()

    publisher = create_ingest_quality_publisher(
        settings,
        extras,
        producer_factory=factory,
    )

    assert publisher is not None
    assert isinstance(publisher, KafkaIngestQualityPublisher)
    assert publisher.topic == "telemetry.ingest.quality"
    assert publisher.key == "quality"
    assert publisher.flush_timeout is None
    assert created["config"]["bootstrap.servers"] == "localhost:9092"


def test_create_ingest_quality_publisher_disabled(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    extras = {
        "KAFKA_INGEST_QUALITY_TOPIC": "telemetry.ingest.quality",
        "KAFKA_INGEST_QUALITY_ENABLED": "false",
    }

    with caplog.at_level("INFO"):
        publisher = create_ingest_quality_publisher(
            settings,
            extras,
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("quality publisher disabled" in message for message in caplog.messages)


def test_create_ingest_quality_publisher_requires_topic(caplog) -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with caplog.at_level("WARNING"):
        publisher = create_ingest_quality_publisher(
            settings,
            {},
            producer_factory=lambda config: _FakeProducer(),
        )

    assert publisher is None
    assert any("quality publisher skipped" in message.lower() for message in caplog.messages)


def test_resolve_ingest_topic_specs_with_overrides() -> None:
    mapping = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_DEFAULT_TOPIC": "timescale.default",
        "KAFKA_INGEST_TOPIC_PARTITIONS": "3",
        "KAFKA_INGEST_TOPIC_REPLICATION_FACTOR": "2",
        "KAFKA_INGEST_TOPIC_DAILY_BARS_PARTITIONS": "5",
        "KAFKA_INGEST_TOPIC_CONFIG": json.dumps({"cleanup.policy": "compact"}),
        "KAFKA_INGEST_TOPIC_DAILY_BARS_CONFIG": json.dumps({"retention.ms": 3_600_000}),
    }

    specs = resolve_ingest_topic_specs(mapping)
    names = [spec.name for spec in specs]
    assert names == ["timescale.daily", "timescale.default"]

    daily = specs[0]
    assert daily.partitions == 5
    assert daily.replication_factor == 2
    assert daily.config == {"cleanup.policy": "compact", "retention.ms": "3600000"}

    default = specs[1]
    assert default.partitions == 3
    assert default.replication_factor == 2
    assert default.config == {"cleanup.policy": "compact"}


def test_should_auto_create_topics_recognises_flags() -> None:
    assert should_auto_create_topics({}) is False
    assert should_auto_create_topics({"KAFKA_AUTO_CREATE_TOPICS": "yes"}) is True
    assert should_auto_create_topics({"KAFKA_INGEST_AUTO_CREATE_TOPICS": "1"}) is True


def test_kafka_topic_provisioner_creates_missing_topics() -> None:
    settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    admin = _FakeAdminClient(existing=("timescale.daily",))
    provisioner = KafkaTopicProvisioner(
        settings,
        admin_factory=lambda config: admin,
        new_topic_factory=lambda spec: spec,
    )

    specs = [
        KafkaTopicSpec(name="timescale.daily", partitions=3, replication_factor=1),
        KafkaTopicSpec(
            name="timescale.intraday",
            partitions=4,
            replication_factor=2,
            config={"cleanup.policy": "compact"},
        ),
    ]

    summary = provisioner.ensure_topics(specs)

    assert summary.requested == ("timescale.daily", "timescale.intraday")
    assert summary.existing == ("timescale.daily",)
    assert summary.created == ("timescale.intraday",)
    assert summary.failed == {}
    assert admin.list_calls == 1
    assert admin.create_calls == 1
    assert admin.created[0].name == "timescale.intraday"
    assert admin.created[0].partitions == 4
    assert admin.created[0].replication_factor == 2
    assert admin.created[0].config == {"cleanup.policy": "compact"}


def test_kafka_topic_provisioner_skips_when_unconfigured() -> None:
    settings = KafkaConnectionSettings.from_mapping({})
    provisioner = KafkaTopicProvisioner(
        settings,
        admin_factory=lambda config: _FakeAdminClient(),
        new_topic_factory=lambda spec: spec,
    )

    summary = provisioner.ensure_topics([KafkaTopicSpec(name="timescale.daily")])

    assert summary.created == ()
    assert summary.existing == ()
    assert summary.requested == ("timescale.daily",)
    assert any("skipping" in note or "not configured" in note for note in summary.notes)


def test_backfill_ingest_dimension_to_kafka(tmp_path: Path) -> None:
    db_path = tmp_path / "timescale.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")

    engine = settings.create_engine()
    try:
        TimescaleMigrator(engine).apply()
        ingestor = TimescaleIngestor(engine)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2024, 1, 2, tzinfo=UTC),
                    "symbol": "EURUSD",
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.05,
                    "adj_close": 1.04,
                    "volume": 1000,
                },
                {
                    "timestamp": datetime(2024, 1, 3, tzinfo=UTC),
                    "symbol": "EURUSD",
                    "open": 1.05,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.18,
                    "adj_close": 1.17,
                    "volume": 1200,
                },
            ]
        )
        ingestor.upsert_daily_bars(frame, source="seed")
    finally:
        engine.dispose()

    kafka_settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})
    extras = {
        "KAFKA_INGEST_TOPICS": "daily_bars:timescale.daily",
        "KAFKA_INGEST_FLUSH_TIMEOUT": "off",
    }
    producer = _FakeProducer()

    summary = backfill_ingest_dimension_to_kafka(
        dimension="daily_bars",
        timescale_settings=settings,
        kafka_settings=kafka_settings,
        mapping=extras,
        identifiers=["EURUSD"],
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 3, tzinfo=UTC),
        producer_factory=lambda config: producer,
    )

    assert isinstance(summary, KafkaIngestBackfillSummary)
    assert summary.dimension == "daily_bars"
    assert summary.topic == "timescale.daily"
    assert summary.rows == 2
    assert summary.symbols == ("EURUSD",)
    assert summary.start_ts is not None
    assert summary.end_ts is not None

    assert len(producer.messages) == 1
    topic, value, key = producer.messages[0]
    assert topic == "timescale.daily"
    assert key == "EURUSD"
    payload = json.loads(value.decode("utf-8"))
    assert payload["result"]["rows_written"] == 2
    assert payload["metadata"]["backfill"] is True
    assert (
        payload["metadata"]["requested_filters"]["start"]
        == datetime(2024, 1, 2, tzinfo=UTC).isoformat()
    )
    assert (
        payload["metadata"]["requested_filters"]["end"]
        == datetime(2024, 1, 3, tzinfo=UTC).isoformat()
    )


def test_backfill_ingest_dimension_requires_topic(tmp_path: Path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'empty.db'}")
    kafka_settings = KafkaConnectionSettings.from_mapping({"KAFKA_BROKERS": "localhost:9092"})

    with pytest.raises(ValueError):
        backfill_ingest_dimension_to_kafka(
            dimension="daily_bars",
            timescale_settings=settings,
            kafka_settings=kafka_settings,
            mapping={},
            producer_factory=lambda config: _FakeProducer(),
        )
