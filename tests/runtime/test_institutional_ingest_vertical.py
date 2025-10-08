from __future__ import annotations

import asyncio
from collections import deque
import dataclasses
from datetime import UTC, datetime
from typing import Callable

import pytest

from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    RedisCachePolicy,
    RedisConnectionSettings,
)
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
    plan_managed_manifest,
)
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
)
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


_EXPECTED_POLICY_METADATA = {
    "ttl_seconds": 900,
    "max_keys": 1024,
    "namespace": "emp:cache",
    "invalidate_prefixes": [],
}


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
        redis_settings=RedisConnectionSettings.from_mapping(
            {"REDIS_URL": "redis://localhost:6379/0"}
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
        redis_policy=config.redis_policy,
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

    assert services.redis_cache is not None
    services.redis_cache.set("timescale:daily:EURUSD", {"price": 1.1})
    services.redis_cache.get("timescale:daily:EURUSD")
    services.redis_cache.get("timescale:daily:GBPUSD")

    await services.stop()
    assert supervisor.active_count == 0
    assert not services.scheduler.running

    summary = services.summary()
    assert summary["timescale"]["configured"] is True
    assert summary["redis"] is not None
    assert summary["redis_policy"] == _EXPECTED_POLICY_METADATA
    assert summary["kafka_topics"] == ["telemetry.ingest"]
    kafka_metadata = summary["kafka_metadata"]
    assert kafka_metadata["timescale_dimensions"] == []
    assert kafka_metadata["consumer_group"] == "emp-ingest-bridge"
    assert kafka_metadata["provisioned"] is True
    assert kafka_metadata["consumer_topics_configured"] is True
    assert kafka_metadata["configured_topics"] == ("telemetry.ingest",)
    assert kafka_metadata["poll_timeout_seconds"] == pytest.approx(1.0)
    assert kafka_metadata["idle_sleep_seconds"] == pytest.approx(0.01)
    redis_metrics = summary["redis_metrics"]
    assert redis_metrics is not None
    assert redis_metrics["namespace"] == "emp:cache"
    assert redis_metrics["hits"] == 1
    assert redis_metrics["misses"] == 1
    assert redis_metrics["requests"] == 2
    assert redis_metrics["hit_rate"] == pytest.approx(0.5)
    manifest = summary["managed_manifest"]
    assert {entry["name"] for entry in manifest} == {"timescale", "redis", "kafka"}
    for entry in manifest:
        if entry["name"] == "redis" and entry["metadata"]["summary"]:
            assert "***" in entry["metadata"]["summary"]
            assert entry["metadata"]["policy"] == _EXPECTED_POLICY_METADATA
            assert entry["metadata"]["metrics"] == redis_metrics


def test_provision_respects_custom_redis_policy() -> None:
    config = _ingest_config()
    custom_policy = RedisCachePolicy(
        ttl_seconds=30,
        max_keys=64,
        namespace="emp:custom",
        invalidate_prefixes=("timescale:daily",),
    )
    config = dataclasses.replace(config, redis_policy=custom_policy)

    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings.from_mapping(
            {"REDIS_URL": "redis://localhost:6379/0"}
        ),
        redis_policy=config.redis_policy,
    )

    services = provisioner.provision(
        run_ingest=lambda: True,
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="custom-policy"),
        redis_client_factory=lambda *_: InMemoryRedis(),
    )

    assert services.redis_policy is custom_policy
    assert services.redis_cache is not None
    assert services.redis_cache.policy is custom_policy

    summary = services.summary()
    policy_snapshot = summary["redis_policy"]
    assert policy_snapshot == {
        "ttl_seconds": 30,
        "max_keys": 64,
        "namespace": "emp:custom",
        "invalidate_prefixes": ["timescale:daily"],
    }
    metrics_snapshot = summary["redis_metrics"]
    assert metrics_snapshot is not None
    assert metrics_snapshot["namespace"] == "emp:custom"
    assert metrics_snapshot["hit_rate"] is None


def test_provision_configures_redis_when_factory_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _ingest_config()
    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://cache.example.com:6379/0"}
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=redis_settings,
        redis_policy=config.redis_policy,
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
    assert services.redis_policy is provisioner.redis_policy
    assert captured == {
        "settings": redis_settings,
        "factory": None,
        "ping": True,
    }


def test_summary_includes_failover_metadata() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_policy=config.redis_policy,
    )
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
    assert services.summary()["redis_policy"] is not None
    assert services.summary()["redis_metrics"] is None


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
        redis_policy=config.redis_policy,
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
    assert summary["redis_policy"] == _EXPECTED_POLICY_METADATA
    assert summary["redis_metrics"] is None


@pytest.mark.asyncio
async def test_connectivity_report_uses_optional_probes() -> None:
    config = _ingest_config(schedule=default_institutional_schedule())
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=RedisConnectionSettings.from_mapping(
            {"REDIS_URL": "redis://localhost:6379/0"}
        ),
        redis_policy=config.redis_policy,
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
    assert manifest["redis"].metadata["policy"] == _EXPECTED_POLICY_METADATA
    assert "metrics" in manifest["redis"].metadata
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
        redis_policy=config.redis_policy,
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
    assert manifest["redis"].metadata["policy"] == _EXPECTED_POLICY_METADATA
    assert "metrics" in manifest["redis"].metadata

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
        redis_policy=config.redis_policy,
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
    redis_snapshot = next(snap for snap in snapshots if snap.name == "redis")
    assert redis_snapshot.metadata["policy"] == _EXPECTED_POLICY_METADATA
    assert "metrics" in redis_snapshot.metadata


@pytest.mark.asyncio
async def test_run_failover_drill_attaches_managed_metadata() -> None:
    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3)
    config = _ingest_config(schedule=schedule)
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=("EURUSD", "GBPUSD")),
        intraday=None,
        macro=None,
    )
    config = dataclasses.replace(config, plan=plan)

    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://localhost:6379/0"}
    )
    provisioner = InstitutionalIngestProvisioner(
        config,
        redis_settings=redis_settings,
        redis_policy=config.redis_policy,
    )

    services = provisioner.provision(
        run_ingest=lambda: asyncio.sleep(0),
        event_bus=_DummyEventBus(),
        task_supervisor=TaskSupervisor(namespace="failover-drill"),
        redis_client_factory=lambda *_: InMemoryRedis(),
    )

    now = datetime.now(tz=UTC)
    results = {
        "daily_bars": TimescaleIngestResult(
            24,
            ("EURUSD", "GBPUSD"),
            now,
            now,
            1.2,
            45.0,
            "daily_bars",
            "yahoo",
        )
    }

    fallback_called = False

    async def _fallback() -> None:
        nonlocal fallback_called
        fallback_called = True

    snapshot = await services.run_failover_drill(
        results,
        fail_dimensions=("daily_bars",),
        fallback=_fallback,
    )

    assert snapshot.scenario == config.failover_drill.label
    assert snapshot.failover_decision.should_failover is True
    assert snapshot.metadata["requested_dimensions"] == ["daily_bars"]
    assert snapshot.metadata["failover_drill"]["label"] == config.failover_drill.label
    manifest = snapshot.metadata.get("managed_manifest")
    assert isinstance(manifest, list) and manifest
    fallback_metadata = snapshot.metadata.get("fallback")
    assert fallback_metadata is not None
    assert fallback_metadata["executed"] is fallback_called

    await services.stop()


def test_plan_managed_manifest_uses_configuration() -> None:
    config = _ingest_config()
    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://cache.example.com:6379/1"}
    )
    kafka_mapping = {
        "KAFKA_BROKERS": "broker:9092",
        "KAFKA_INGEST_CONSUMER_TOPICS": "telemetry.ingest, telemetry.drills",
    }

    manifest = plan_managed_manifest(
        config,
        redis_settings=redis_settings,
        kafka_mapping=kafka_mapping,
    )

    snapshots = {snapshot.name: snapshot for snapshot in manifest}
    timescale_metadata = snapshots["timescale"].metadata
    assert timescale_metadata["application_name"] == config.timescale_settings.application_name
    assert "***" in timescale_metadata["url"]
    assert snapshots["redis"].metadata["summary"].startswith("Redis")
    kafka_metadata = snapshots["kafka"].metadata
    assert kafka_metadata["bootstrap_servers"] == "broker:9092"
    assert set(kafka_metadata["topics"]) == {"telemetry.ingest", "telemetry.drills"}
    assert kafka_metadata["consumer_topics_configured"] is True
    assert kafka_metadata["consumer_group"] == "emp-ingest-bridge"
    assert kafka_metadata["provisioned"] is False
    assert kafka_metadata["configured_topics"] == ("telemetry.drills", "telemetry.ingest")
    assert kafka_metadata["topic_count"] == 2
    assert snapshots["redis"].metadata["policy"] == _EXPECTED_POLICY_METADATA
    assert "metrics" in snapshots["redis"].metadata


def test_plan_managed_manifest_adds_default_topic_when_missing() -> None:
    config = _ingest_config()

    manifest = plan_managed_manifest(config, kafka_mapping={})
    kafka_metadata = next(snapshot for snapshot in manifest if snapshot.name == "kafka").metadata
    assert "telemetry.ingest" in kafka_metadata["topics"]
    assert kafka_metadata["consumer_topics_configured"] is True
    assert kafka_metadata["configured_topics"] == ("telemetry.ingest",)
    assert kafka_metadata["topic_count"] == 1
    redis_metadata = next(snapshot for snapshot in manifest if snapshot.name == "redis").metadata
    assert redis_metadata["policy"] == _EXPECTED_POLICY_METADATA
    assert "metrics" in redis_metadata
