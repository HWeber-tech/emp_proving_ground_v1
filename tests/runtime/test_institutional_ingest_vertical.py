from __future__ import annotations

import asyncio
from collections import deque
import dataclasses

import pytest

from src.data_foundation.cache.redis_cache import InMemoryRedis, RedisConnectionSettings
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    KafkaReadinessSettings,
    TimescaleBackupSettings,
    TimescaleFailoverDrillSettings,
    TimescaleIngestRecoverySettings,
    TimescaleRetentionSettings,
)
from src.data_foundation.ingest.institutional_vertical import (
    InstitutionalIngestProvisioner,
    default_institutional_schedule,
)
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import TimescaleBackbonePlan
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings
from src.runtime.task_supervisor import TaskSupervisor


class _DummyEventBus:
    def __init__(self) -> None:
        self.published: deque[tuple[str, dict[str, object]]] = deque()

    def publish_from_sync(self, event) -> None:  # pragma: no cover - defensive path
        self.published.append((event.type, dict(event.payload)))

    def is_running(self) -> bool:
        return True


class _DummyKafkaConsumer:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.closed = False

    def subscribe(self, topics) -> None:
        self.subscribed = list(topics)

    def poll(self, timeout):
        return None

    def close(self) -> None:
        self.closed = True


def _ingest_config(*, schedule: IngestSchedule | None = None) -> InstitutionalIngestConfig:
    return InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(),
        timescale_settings=TimescaleConnectionSettings(
            url="postgresql://user:secret@example.com/emp"
        ),
        kafka_settings=KafkaConnectionSettings.from_mapping(
            {"KAFKA_BROKERS": "broker:9092"}
        ),
        metadata={},
        schedule=schedule,
        recovery=TimescaleIngestRecoverySettings(),
        operational_alert_routes={},
        backup=TimescaleBackupSettings(enabled=False),
        retention=TimescaleRetentionSettings(),
        spark_export=None,
        failover_drill=TimescaleFailoverDrillSettings(
            enabled=True, dimensions=("daily",)
        ),
        spark_stress=None,
        cross_region=None,
        kafka_readiness=KafkaReadinessSettings(enabled=True),
    )


@pytest.mark.asyncio
async def test_provisioned_services_supervise_components() -> None:
    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3)
    config = _ingest_config(schedule=schedule)
    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://localhost:6379/0"}
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=redis_settings,
        kafka_mapping={
            "KAFKA_INGEST_CONSUMER_TOPICS": "telemetry.ingest",
            "KAFKA_INGEST_CONSUMER_POLL_TIMEOUT": "0",
            "KAFKA_INGEST_CONSUMER_IDLE_SLEEP": "0.01",
        },
    )

    run_calls: list[int] = []

    async def _run_ingest() -> bool:
        run_calls.append(1)
        return True

    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="test")

    services = provisioner.provision(
        run_ingest=_run_ingest,
        event_bus=bus,
        task_supervisor=supervisor,
        redis_client_factory=lambda settings: InMemoryRedis(),
        kafka_consumer_factory=lambda config: _DummyKafkaConsumer(),
    )

    assert services.redis_cache is not None
    assert services.kafka_consumer is not None

    services.start()
    await asyncio.sleep(0.1)
    assert run_calls

    await services.stop()
    assert supervisor.active_count == 0
    assert not services.scheduler.running

    summary = services.summary()
    assert summary["timescale"]["configured"] is True
    assert summary["redis"] is not None
    assert summary["kafka_topics"] == ["telemetry.ingest"]


def test_summary_includes_failover_metadata() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    provisioner = InstitutionalIngestProvisioner(config)
    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="summary"),
    )

    metadata = services.failover_metadata()
    assert metadata == {
        "enabled": True,
        "dimensions": ["daily"],
        "label": "required_timescale_failover",
        "run_fallback": True,
    }


@pytest.mark.asyncio
async def test_provision_skips_optional_connectors_when_unconfigured() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    config = dataclasses.replace(
        config,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        failover_drill=None,
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings(),
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="optional"),
    )

    assert services.redis_cache is None
    assert services.kafka_consumer is None

    services.start()
    await services.stop()

    summary = services.summary()
    assert summary["redis"] is None
    assert summary["kafka"] is None
