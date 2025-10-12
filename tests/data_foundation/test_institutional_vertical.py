"""Regression and coverage guardrails for institutional ingest services."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from typing import Any, Coroutine, Mapping

import pytest


pytestmark = pytest.mark.guardrail

from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
)
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    TimescaleFailoverDrillSettings,
)
from src.data_foundation.ingest.institutional_vertical import (
    ConnectivityProbeError,
    InstitutionalIngestServices,
    InstitutionalIngestProvisioner,
    ManagedConnectorSnapshot,
    _extract_kafka_topics,
    _plan_dimensions,
    _policy_metadata,
    _redacted_url,
    default_institutional_schedule,
    plan_managed_manifest,
)
from src.data_foundation.ingest.scheduler import IngestSchedule, TimescaleIngestScheduler
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
)
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings
from src.runtime.task_supervisor import TaskSupervisor


def _make_services() -> InstitutionalIngestServices:
    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(),
        timescale_settings=TimescaleConnectionSettings(url="sqlite:///tmp/test.db"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers=""),
        redis_settings=RedisConnectionSettings(),
    )
    scheduler = Mock(spec=TimescaleIngestScheduler)
    supervisor = Mock(spec=TaskSupervisor)
    return InstitutionalIngestServices(
        config=config,
        scheduler=scheduler,
        task_supervisor=supervisor,
    )


class _RecordingScheduler:
    def __init__(self) -> None:
        self.running = False
        self.started = 0
        self.stopped = 0
        self._state = SimpleNamespace(
            interval_seconds=1800.0,
            jitter_seconds=60.0,
            max_failures=3,
            next_run_at=None,
        )

    def start(self) -> None:
        self.running = True
        self.started += 1

    async def stop(self) -> None:
        self.running = False
        self.stopped += 1

    def state(self) -> SimpleNamespace:
        return self._state


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str,
        metadata: Mapping[str, object],
    ) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.calls.append({"name": name, "metadata": dict(metadata)})
        return task


class _RecordingKafkaConsumer:
    def __init__(self) -> None:
        self.topics = ("timescale.daily", "timescale.intraday")
        self.pings: int = 0

    def run_forever(self, stop_event: asyncio.Event) -> Coroutine[Any, Any, None]:
        async def _runner() -> None:
            await stop_event.wait()

        return _runner()

    def ping(self) -> bool:
        self.pings += 1
        return True


def test_timescale_probe_expected_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    services = _make_services()
    caplog.set_level(logging.WARNING)

    def raise_ose(_: TimescaleConnectionSettings) -> None:
        raise OSError("engine boom")

    monkeypatch.setattr(TimescaleConnectionSettings, "create_engine", raise_ose)

    probe = services._default_connectivity_probes()["timescale"]
    with pytest.raises(ConnectivityProbeError):
        probe()
    assert "timescale connectivity probe failed" in caplog.text


def test_redis_probe_expected_failure(caplog: pytest.LogCaptureFixture) -> None:
    services = _make_services()
    caplog.set_level(logging.WARNING)

    class FailingRedisCache:
        def __init__(self) -> None:
            self.raw_client = self

        def ping(self) -> bool:
            raise TimeoutError("redis ping timeout")

    services.redis_cache = FailingRedisCache()

    probe = services._default_connectivity_probes()["redis"]
    with pytest.raises(ConnectivityProbeError):
        probe()
    assert "redis connectivity probe failed" in caplog.text


@pytest.mark.asyncio
async def test_connectivity_report_marks_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()

    snapshot = ManagedConnectorSnapshot(
        name="timescale",
        configured=True,
        supervised=True,
        metadata={},
    )
    def manifest(_: InstitutionalIngestServices) -> tuple[ManagedConnectorSnapshot, ...]:
        return (snapshot,)

    monkeypatch.setattr(InstitutionalIngestServices, "managed_manifest", manifest)

    def failing_probe() -> bool:
        raise ConnectivityProbeError("expected failure")

    result = await services.connectivity_report(probes={"timescale": failing_probe})
    assert result[0].healthy is False
    assert result[0].error == "expected failure"


@pytest.mark.asyncio
async def test_connectivity_report_propagates_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()

    snapshot = ManagedConnectorSnapshot(
        name="timescale",
        configured=True,
        supervised=True,
        metadata={},
    )
    def manifest(_: InstitutionalIngestServices) -> tuple[ManagedConnectorSnapshot, ...]:
        return (snapshot,)

    monkeypatch.setattr(InstitutionalIngestServices, "managed_manifest", manifest)

    def unexpected_probe() -> bool:
        raise RuntimeError("probe exploded")

    with pytest.raises(RuntimeError):
        await services.connectivity_report(probes={"timescale": unexpected_probe})


def test_kafka_topic_extraction_and_plan_helpers() -> None:
    mapping = {
        "KAFKA_INGEST_TOPICS": "daily:timescale.daily, intraday: timescale.intraday",
        "KAFKA_INGEST_DEFAULT_TOPIC": " defaults.ingest ",
        "KAFKA_INGEST_CONSUMER_TOPICS": "alpha , beta ",
    }
    topics = _extract_kafka_topics(mapping)
    assert topics == (
        "alpha",
        "beta",
        "defaults.ingest",
        "timescale.daily",
        "timescale.intraday",
    )

    schedule = default_institutional_schedule()
    assert isinstance(schedule, IngestSchedule)
    assert schedule.interval_seconds > 0
    assert schedule.jitter_seconds > 0

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=("EURUSD",)),
        intraday=IntradayTradeIngestPlan(symbols=("EURUSD",), interval="5m"),
        macro=MacroEventIngestPlan(start="2024-01-01", end="2024-01-10"),
    )
    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=plan,
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
    )
    assert _plan_dimensions(config) == ("daily", "intraday", "macro")


def test_redacted_url_and_policy_metadata() -> None:
    url = "postgresql://svc-user:S3cret!@db.example.com:5432/ingest"
    redacted = _redacted_url(url)
    assert "S3cret" not in redacted
    assert redacted.startswith("postgresql://svc-user")

    policy = RedisCachePolicy(ttl_seconds=30, max_keys=42, namespace="emp:test", invalidate_prefixes=("timescale",))
    metadata = _policy_metadata(policy)
    assert metadata == {
        "ttl_seconds": 30,
        "max_keys": 42,
        "namespace": "emp:test",
        "invalidate_prefixes": ["timescale"],
    }


@pytest.mark.asyncio
async def test_services_start_stop_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _RecordingScheduler()
    supervisor = _RecordingSupervisor()
    kafka_consumer = _RecordingKafkaConsumer()

    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD", "GBPUSD")),
            intraday=IntradayTradeIngestPlan(symbols=("EURUSD",)),
        ),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/1"),
        failover_drill=TimescaleFailoverDrillSettings(enabled=True, dimensions=("daily",)),
        metadata={"kafka_streaming_enabled": True},
    )

    services = InstitutionalIngestServices(
        config=config,
        scheduler=scheduler,
        task_supervisor=supervisor,
        redis_settings=config.redis_settings,
        redis_cache=ManagedRedisCache(InMemoryRedis(), config.redis_policy),
        redis_policy=config.redis_policy,
        kafka_settings=config.kafka_settings,
        kafka_consumer=kafka_consumer,
        kafka_metadata={
            "bridge": "enabled",
            "streaming_enabled": True,
            "streaming_active": True,
        },
    )

    services.start()
    assert scheduler.running is True
    assert supervisor.calls[0]["metadata"]["component"] == "timescale_ingest.kafka_consumer"

    summary = services.summary()
    assert summary["timescale"]["url"].startswith("postgresql://svc")
    assert summary["redis_backing"] == "InMemoryRedis"
    assert summary["kafka_topics"] == list(kafka_consumer.topics)
    assert summary["redis_policy"]["ttl_seconds"] == config.redis_policy.ttl_seconds
    assert summary["kafka_streaming_enabled"] is True
    kafka_meta = summary["kafka_metadata"]
    assert kafka_meta["streaming_enabled"] is True
    assert kafka_meta["streaming_active"] is True

    manifest = services.managed_manifest()
    assert len(manifest) == 3
    timescale_snapshot = next(snapshot for snapshot in manifest if snapshot.name == "timescale")
    assert timescale_snapshot.supervised is True

    await services.stop()
    assert scheduler.running is False


def test_services_summary_uses_configured_topics_when_consumer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop() -> bool:
        return True

    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
    )

    provisioner = InstitutionalIngestProvisioner(
        ingest_config=config,
        kafka_mapping={
            "KAFKA_INGEST_CONSUMER_TOPICS": "telemetry.ingest,telemetry.drills",
        },
    )

    monkeypatch.setattr(
        "src.data_foundation.ingest.institutional_vertical.create_ingest_event_consumer",
        lambda *_, **__: None,
    )

    services = provisioner.provision(
        run_ingest=_noop,
        event_bus=object(),
        task_supervisor=_RecordingSupervisor(),
        redis_client_factory=lambda *_: InMemoryRedis(),
    )

    summary = services.summary()
    kafka_metadata = summary["kafka_metadata"]
    assert kafka_metadata["provisioned"] is False
    assert kafka_metadata["consumer_topics_configured"] is True
    assert kafka_metadata["configured_topics"] == (
        "telemetry.drills",
        "telemetry.ingest",
    )
    assert summary["kafka_topics"] == ["telemetry.drills", "telemetry.ingest"]
    assert kafka_metadata["streaming_enabled"] is True
    assert kafka_metadata["streaming_active"] is False


@pytest.mark.asyncio
async def test_services_connectivity_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    services = _make_services()
    manifest = (
        ManagedConnectorSnapshot(
            name="timescale",
            configured=True,
            supervised=True,
            metadata={},
        ),
    )
    monkeypatch.setattr(
        InstitutionalIngestServices,
        "managed_manifest",
        lambda self: manifest,
    )

    results = await services.connectivity_report(probes={"timescale": lambda: True})
    assert results[0].healthy is True
    assert results[0].error is None


@pytest.mark.asyncio
async def test_services_run_failover_drill_builds_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _RecordingScheduler()
    supervisor = _RecordingSupervisor()

    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD",)),
            intraday=IntradayTradeIngestPlan(symbols=("EURUSD",)),
        ),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
        failover_drill=TimescaleFailoverDrillSettings(enabled=True, dimensions=(), label="custom-drill"),
    )

    services = InstitutionalIngestServices(
        config=config,
        scheduler=scheduler,
        task_supervisor=supervisor,
        redis_settings=config.redis_settings,
        redis_cache=None,
        redis_policy=config.redis_policy,
        kafka_settings=config.kafka_settings,
        kafka_consumer=None,
    )

    captured: dict[str, object] = {}

    async def fake_execute_failover_drill(**kwargs: object) -> str:  # type: ignore[override]
        captured.update(kwargs)
        return "drill-complete"

    monkeypatch.setattr(
        "src.data_foundation.ingest.institutional_vertical.execute_failover_drill",
        fake_execute_failover_drill,
    )

    results = {
        "daily": TimescaleIngestResult.empty(dimension="daily", source="test"),
        "intraday": TimescaleIngestResult.empty(dimension="intraday", source="test"),
    }

    outcome = await services.run_failover_drill(results, metadata={"invoker": "pytest"})
    assert outcome == "drill-complete"
    assert captured["scenario"] == "custom-drill"
    assert captured["metadata"]["requested_dimensions"]
    assert captured["metadata"]["invoker"] == "pytest"


@pytest.mark.asyncio
async def test_services_stop_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _RecordingScheduler()
    supervisor = _RecordingSupervisor()
    services = InstitutionalIngestServices(
        config=InstitutionalIngestConfig(
            should_run=True,
            reason=None,
            plan=TimescaleBackbonePlan(),
            timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
            kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
            redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
        ),
        scheduler=scheduler,
        task_supervisor=supervisor,
    )

    stop_event = asyncio.Event()

    async def _never() -> None:
        await asyncio.sleep(10)

    services._kafka_stop_event = stop_event
    services._kafka_task = asyncio.create_task(_never())

    async def fake_wait_for(_awaitable: asyncio.Task, timeout: float) -> object:
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    await services.stop()
    assert services._kafka_task is None
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_services_run_failover_drill_validates_inputs() -> None:
    drill_enabled_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD",)),
        ),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
        failover_drill=TimescaleFailoverDrillSettings(enabled=True, dimensions=("daily",)),
    )
    services = InstitutionalIngestServices(
        config=drill_enabled_config,
        scheduler=_RecordingScheduler(),
        task_supervisor=_RecordingSupervisor(),
    )

    with pytest.raises(ValueError):
        await services.run_failover_drill({})

    services_disabled = InstitutionalIngestServices(
        config=InstitutionalIngestConfig(
            should_run=True,
            reason=None,
            plan=TimescaleBackbonePlan(),
            timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
            kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
            redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
        ),
        scheduler=_RecordingScheduler(),
        task_supervisor=_RecordingSupervisor(),
    )

    with pytest.raises(RuntimeError):
        await services_disabled.run_failover_drill(
            {"daily": TimescaleIngestResult.empty(dimension="daily")}
        )


@pytest.mark.asyncio
async def test_services_connectivity_report_accepts_async_probe() -> None:
    services = _make_services()
    async def async_probe() -> bool:
        await asyncio.sleep(0)
        return True

    manifest = services.managed_manifest()
    probes = {snapshot.name: async_probe for snapshot in manifest}
    summary = await services.connectivity_report(probes=probes)
    assert summary[0].healthy is True


def test_plan_managed_manifest_resolves_kafka_mapping() -> None:
    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=("EURUSD",))),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
    )

    manifest = plan_managed_manifest(
        config,
        kafka_mapping={
            "KAFKA_INGEST_TOPICS": "intraday:timescale.intraday",
            "KAFKA_INGEST_DEFAULT_TOPIC": "timescale.default",
        },
    )

    kafka_snapshot = next(snapshot for snapshot in manifest if snapshot.name == "kafka")
    assert kafka_snapshot.metadata["topics"]
    assert kafka_snapshot.metadata["consumer_topics_configured"] is True
    assert kafka_snapshot.metadata["consumer_group"] == "emp-ingest-bridge"
    assert kafka_snapshot.metadata["configured_topics"] == (
        "timescale.default",
        "timescale.intraday",
    )
    assert kafka_snapshot.metadata["topic_count"] == 2
    assert kafka_snapshot.metadata["streaming_enabled"] is True
    assert kafka_snapshot.metadata["streaming_active"] is False


def test_provisioner_builds_services(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=("EURUSD",))),
        timescale_settings=TimescaleConnectionSettings(url="postgresql://svc:secret@db/ingest"),
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="broker:9092"),
        redis_settings=RedisConnectionSettings(url="redis://cache:6379/0"),
    )

    provisioner = InstitutionalIngestProvisioner(
        ingest_config=config,
        kafka_mapping={"KAFKA_INGEST_DEFAULT_TOPIC": "timescale.default"},
    )

    fake_consumer = _RecordingKafkaConsumer()

    def consumer_factory(config: Mapping[str, object]) -> _RecordingKafkaConsumer:
        assert "bootstrap.servers" in config
        return fake_consumer

    monkeypatch.setattr(
        "src.data_foundation.ingest.institutional_vertical.create_ingest_event_consumer",
        lambda *args, **kwargs: fake_consumer,
    )

    async def dummy_run_ingest() -> None:
        return None

    services = provisioner.provision(
        run_ingest=dummy_run_ingest,
        event_bus=object(),
        task_supervisor=_RecordingSupervisor(),
        redis_client_factory=lambda settings: InMemoryRedis(),
        kafka_consumer_factory=consumer_factory,
    )

    assert isinstance(services.scheduler, TimescaleIngestScheduler)
    assert services.redis_cache is not None
    assert services.kafka_consumer is fake_consumer
