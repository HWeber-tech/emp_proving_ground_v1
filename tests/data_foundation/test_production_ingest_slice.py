import asyncio
import dataclasses
from collections import deque

import pytest


pytestmark = pytest.mark.guardrail

from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    RedisCachePolicy,
    RedisConnectionSettings,
)
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    TimescaleBackbonePlan,
)
from src.data_foundation.ingest.production_slice import ProductionIngestSlice
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import DailyBarIngestPlan
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


def _ingest_config(*, schedule: IngestSchedule | None = None) -> InstitutionalIngestConfig:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=5)
    )
    return InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=plan,
        timescale_settings=TimescaleConnectionSettings(url="sqlite:///ingest_test.db"),
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        metadata={},
        schedule=schedule,
    )


@pytest.mark.asyncio
async def test_production_slice_runs_once_and_records_summary() -> None:
    config = _ingest_config()
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-test")

    calls: list[TimescaleBackbonePlan] = []

    class _StubOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            calls.append(plan)
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    orchestrator = _StubOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
    )

    success = await slice_runtime.run_once()

    assert success is True
    assert calls == [config.plan]

    summary = slice_runtime.summary()
    assert summary["last_results"]["daily_bars"]["dimension"] == "daily_bars"
    assert summary["last_error"] is None
    assert summary["last_run_at"] is not None


@pytest.mark.asyncio
async def test_production_slice_supervises_scheduler_tasks() -> None:
    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3)
    config = _ingest_config(schedule=schedule)
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-scheduler")

    run_counts: list[int] = []

    class _CountingOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            run_counts.append(1)
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    orchestrator = _CountingOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
    )

    slice_runtime.start()
    await asyncio.sleep(0.12)
    await slice_runtime.stop()

    assert run_counts  # scheduler executed the ingest callback
    assert supervisor.active_count == 0

    summary = slice_runtime.summary()
    assert summary["services"] is not None
    assert summary["services"]["schedule"]["running"] is False


@pytest.mark.asyncio
async def test_production_slice_handles_disabled_configuration() -> None:
    base = _ingest_config()
    disabled = dataclasses.replace(
        base,
        should_run=False,
        reason="Timescale ingest disabled",
        plan=TimescaleBackbonePlan(),
    )

    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-disabled")

    class _FailingOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            raise AssertionError("Ingest should not execute when disabled")

    slice_runtime = ProductionIngestSlice(
        disabled,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: _FailingOrchestrator(),
    )

    success = await slice_runtime.run_once()

    assert success is False
    slice_runtime.start()  # should no-op
    await slice_runtime.stop()

    summary = slice_runtime.summary()
    assert summary["should_run"] is False
    assert summary["services"] is None
    assert summary["last_results"] == {}
    assert summary["last_error"] == "Timescale ingest disabled"


class _StubCache:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def invalidate(self, prefixes: tuple[str, ...] | list[str] | None = None) -> int:
        if prefixes is None:
            resolved: tuple[str, ...] = tuple()
        elif isinstance(prefixes, list):
            resolved = tuple(prefixes)
        else:
            resolved = prefixes
        self.calls.append(resolved)
        return 1


class _StubServices:
    def __init__(self, cache: _StubCache) -> None:
        self.redis_cache = cache

    def start(self) -> None:  # pragma: no cover - not exercised in tests
        return None

    async def stop(self) -> None:  # pragma: no cover - not exercised in tests
        return None

    def summary(self) -> dict[str, object]:  # pragma: no cover - defensive default
        return {}


class _StubProvisioner:
    def __init__(self, services: _StubServices) -> None:
        self._services = services

    def provision(self, *args, **kwargs) -> _StubServices:
        return self._services


@pytest.mark.asyncio
async def test_production_slice_invalidates_redis_cache_after_ingest() -> None:
    config = _ingest_config()
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-invalidations")

    cache = _StubCache()
    services = _StubServices(cache)

    class _StubOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=5,
                    symbols=("EURUSD",),
                    start_ts=None,
                    end_ts=None,
                    ingest_duration_seconds=0.5,
                    freshness_seconds=12.0,
                    dimension="daily_bars",
                    source="yahoo",
                )
            }

    orchestrator = _StubOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
        provisioner_factory=lambda *args, **kwargs: _StubProvisioner(services),
    )

    success = await slice_runtime.run_once()

    assert success is True
    assert cache.calls
    assert cache.calls[-1] == ("timescale:daily_bars",)


@pytest.mark.asyncio
async def test_production_slice_skips_cache_invalidation_when_no_rows() -> None:
    config = _ingest_config()
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-invalidations-skip")

    cache = _StubCache()
    services = _StubServices(cache)

    class _StubOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    orchestrator = _StubOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
        provisioner_factory=lambda *args, **kwargs: _StubProvisioner(services),
    )

    success = await slice_runtime.run_once()

    assert success is True
    assert cache.calls == []


@pytest.mark.asyncio
async def test_production_slice_summary_reflects_custom_redis_policy() -> None:
    config = _ingest_config()
    custom_policy = RedisCachePolicy(
        ttl_seconds=60,
        max_keys=32,
        namespace="emp:test",
        invalidate_prefixes=("timescale:daily",),
    )
    config = dataclasses.replace(config, redis_policy=custom_policy)

    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-policy")
    redis_settings = RedisConnectionSettings.from_mapping(
        {"REDIS_URL": "redis://localhost:6379/0"}
    )

    class _StubOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        redis_settings=redis_settings,
        redis_client_factory=lambda _: InMemoryRedis(),
        orchestrator_factory=lambda settings, publisher: _StubOrchestrator(),
    )

    success = await slice_runtime.run_once()
    assert success is True

    summary = slice_runtime.summary()
    services = summary["services"]
    assert services is not None
    policy_snapshot = services["redis_policy"]
    assert policy_snapshot == {
        "ttl_seconds": 60,
        "max_keys": 32,
        "namespace": "emp:test",
        "invalidate_prefixes": ["timescale:daily"],
    }

    await slice_runtime.stop()
