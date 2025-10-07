from __future__ import annotations

import asyncio
from collections import deque
import dataclasses
from typing import Callable

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
    ManagedConnectorSnapshot,
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

    def list_topics(self, timeout=None):  # type: ignore[override]
        return {"topics": {topic: {} for topic in self.subscribed or ("telemetry.ingest",)}}

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
    manifest = summary["managed_manifest"]
    assert {entry["name"] for entry in manifest} == {"timescale", "redis", "kafka"}
    for entry in manifest:
        if entry["name"] == "redis" and entry["metadata"]["summary"]:
            assert "***" in entry["metadata"]["summary"]


def test_provision_configures_redis_when_factory_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _ingest_config()
    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://cache.example.com:6379/0"}
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=redis_settings,
        kafka_mapping={"KAFKA_INGEST_CONSUMER_TOPICS": "telemetry.ingest"},
    )

    captured: dict[str, object] = {}

    def _fake_configure(
        settings: RedisConnectionSettings,
        *,
        factory: Callable[[RedisConnectionSettings], object] | None = None,
        ping: bool = True,
    ) -> object:
        captured["settings"] = settings
        captured["factory"] = factory
        captured["ping"] = ping
        return InMemoryRedis()

    monkeypatch.setattr(
        "src.data_foundation.ingest.institutional_vertical.configure_redis_client",
        _fake_configure,
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="configure"),
        kafka_consumer_factory=lambda *_: _DummyKafkaConsumer(),
    )

    assert services.redis_cache is not None
    assert isinstance(services.redis_cache.raw_client, InMemoryRedis)
    assert captured == {
        "settings": redis_settings,
        "factory": None,
        "ping": True,
    }


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


@pytest.mark.asyncio
async def test_connectivity_report_uses_optional_probes() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings.from_mapping(
            {"REDIS_URL": "redis://localhost:6379/0"}
        ),
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="probes"),
        redis_client_factory=lambda *_: InMemoryRedis(),
        kafka_consumer_factory=lambda *_: _DummyKafkaConsumer(),
    )

    services.start()
    probes_called: dict[str, bool] = {"timescale": False, "redis": False}

    async def _async_probe() -> bool:
        probes_called["timescale"] = True
        return True

    def _sync_probe() -> bool:
        probes_called["redis"] = True
        return False

    report = await services.connectivity_report(
        probes={"timescale": _async_probe, "redis": _sync_probe},
        timeout=0.1,
    )

    manifest = {snapshot.name: snapshot for snapshot in report}
    assert manifest["timescale"].healthy is True
    assert manifest["redis"].healthy is False
    assert manifest["kafka"].healthy is None
    assert probes_called == {"timescale": True, "redis": True}

    await services.stop()


@pytest.mark.asyncio
async def test_connectivity_report_defaults_use_managed_probes(tmp_path) -> None:
    sqlite_path = tmp_path / "timescale.db"
    config = _ingest_config(schedule=default_institutional_schedule())
    config = dataclasses.replace(
        config,
        timescale_settings=TimescaleConnectionSettings(url=f"sqlite:///{sqlite_path}"),
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings.from_mapping(
            {"REDIS_URL": "redis://localhost:6379/0"}
        ),
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="default-probes"),
        redis_client_factory=lambda *_: InMemoryRedis(),
        kafka_consumer_factory=lambda *_: _DummyKafkaConsumer(),
    )

    report = await services.connectivity_report(timeout=0.1)
    manifest = {snapshot.name: snapshot for snapshot in report}

    assert manifest["timescale"].healthy is True
    assert manifest["redis"].healthy is True
    assert manifest["kafka"].healthy is True

    await services.stop()


def test_manifest_snapshot_structure() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    config = dataclasses.replace(
        config,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings(),
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="manifest"),
    )

    snapshots = services.managed_manifest()
    assert {snap.name for snap in snapshots} == {"timescale", "redis", "kafka"}
    timescale_snapshot = next(snap for snap in snapshots if snap.name == "timescale")
    assert timescale_snapshot.supervised is False
    assert timescale_snapshot.configured is True
    assert "dimensions" in timescale_snapshot.metadata
