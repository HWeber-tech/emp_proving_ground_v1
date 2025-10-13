from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.core.event_bus import Event, EventBus
from src.data_foundation.cache.redis_cache import InMemoryRedis, ManagedRedisCache, RedisCachePolicy
from src.data_foundation.pipelines.backbone_service import OperationalBackboneService
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.in_memory_broker import InMemoryKafkaBroker
from src.data_foundation.streaming.kafka_stream import KafkaIngestEventConsumer, KafkaIngestEventPublisher
from src.data_integration.real_data_integration import RealDataManager
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import DailyBarIngestPlan, TimescaleBackbonePlan
from src.runtime.task_supervisor import TaskSupervisor
from src.sensory.real_sensory_organ import RealSensoryOrgan


try:  # pragma: no cover - optional dependency mirrors production cache wiring
    import fakeredis
except Exception:  # pragma: no cover - fallback when fakeredis unavailable
    fakeredis = None  # type: ignore[assignment]


def _managed_cache() -> ManagedRedisCache:
    policy = RedisCachePolicy.institutional_defaults()
    if fakeredis is not None:
        client = fakeredis.FakeRedis()
    else:
        client = InMemoryRedis()
    return ManagedRedisCache(client, policy)


def _build_consumer(broker: InMemoryKafkaBroker, event_bus: EventBus) -> KafkaIngestEventConsumer:
    consumer = broker.create_consumer()
    return KafkaIngestEventConsumer(
        consumer,
        topics=("telemetry.ingest",),
        event_bus=event_bus,
        poll_timeout=0.05,
        idle_sleep=0.0,
        publish_consumer_lag=False,
    )


def _daily_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": base - timedelta(days=1),
                "symbol": "EURUSD",
                "open": 1.10,
                "high": 1.12,
                "low": 1.08,
                "close": 1.11,
                "adj_close": 1.105,
                "volume": 1200,
            },
            {
                "date": base,
                "symbol": "EURUSD",
                "open": 1.11,
                "high": 1.13,
                "low": 1.09,
                "close": 1.125,
                "adj_close": 1.12,
                "volume": 1500,
            },
        ]
    )


def _intraday_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": base - timedelta(minutes=2),
                "symbol": "EURUSD",
                "price": 1.121,
                "size": 640,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": base - timedelta(minutes=1),
                "symbol": "EURUSD",
                "price": 1.124,
                "size": 720,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )


def _service_fixture(tmp_path) -> tuple[OperationalBackboneService, ManagedRedisCache, InMemoryKafkaBroker, EventBus]:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'service_timescale.db'}")
    broker = InMemoryKafkaBroker()
    publisher = KafkaIngestEventPublisher(
        broker.create_producer(),
        topic_map={"daily_bars": "telemetry.ingest", "intraday_trades": "telemetry.ingest"},
    )
    cache = _managed_cache()
    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
        managed_cache=cache,
    )

    event_bus = EventBus()
    sensory = RealSensoryOrgan(event_bus=event_bus)

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=lambda: _build_consumer(broker, event_bus),
        sensory_organ=sensory,
        event_topics=("telemetry.ingest",),
        shutdown_manager_on_close=False,
    )

    service = OperationalBackboneService(
        manager,
        pipeline,
        owns_manager=True,
        owns_pipeline=True,
    )
    return service, cache, broker, event_bus


@pytest.mark.asyncio()
async def test_operational_backbone_service_ingest_cycle(tmp_path) -> None:
    service, cache, broker, event_bus = _service_fixture(tmp_path)
    base = datetime(2024, 5, 6, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
    )

    metrics: dict[str, object] | None = None
    connectivity = None

    try:
        result = await service.ingest_once(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
            poll_consumer=False,
        )
        metrics = dict(service.cache_metrics())
        connectivity = service.connectivity_report()
    finally:
        await service.shutdown()
        await event_bus.stop()
        flush = getattr(cache.raw_client, "flushall", None)
        if callable(flush):  # pragma: no branch - optional dependency cleanup
            flush()

    assert "daily_bars" in result.ingest_results
    assert result.ingest_results["daily_bars"].rows_written == 2
    assert "intraday_trades" in result.ingest_results
    assert result.ingest_results["intraday_trades"].rows_written == 2

    assert metrics is not None
    assert int(metrics.get("misses", 0)) >= 1

    assert connectivity is not None
    assert connectivity.timescale is True
    assert connectivity.kafka is True
    assert result.task_snapshots
    assert any(
        entry.get("name") == "operational.backbone.ingest" for entry in result.task_snapshots
    )


@pytest.mark.asyncio()
async def test_operational_backbone_service_streaming(tmp_path) -> None:
    service, cache, broker, event_bus = _service_fixture(tmp_path)
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
    )

    received: list[Event] = []
    event_signal = asyncio.Event()

    await event_bus.start()

    def _collect(event: Event) -> None:
        received.append(event)
        event_signal.set()

    subscription = event_bus.subscribe("telemetry.ingest", _collect)

    try:
        await service.ensure_streaming()
        result = await service.ingest_once(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
            poll_consumer=False,
        )

        await asyncio.wait_for(event_signal.wait(), timeout=1.0)
        assert received
        assert service.streaming_active is True
        assert any(
            entry.get("name") == "operational.backbone.ingest"
            for entry in result.task_snapshots
        )
    finally:
        await service.stop_streaming()
        event_bus.unsubscribe(subscription)
        await service.shutdown()
        await event_bus.stop()
        flush = getattr(cache.raw_client, "flushall", None)
        if callable(flush):  # pragma: no branch - optional dependency cleanup
            flush()

    assert service.streaming_active is False


@pytest.mark.asyncio()
async def test_operational_backbone_service_scheduler(tmp_path) -> None:
    service, cache, broker, event_bus = _service_fixture(tmp_path)
    base = datetime(2024, 8, 2, tzinfo=timezone.utc)
    runs = 0

    def _plan() -> TimescaleBackbonePlan:
        return TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD",), lookback_days=1),
        )

    def _fetch_daily(symbols: list[str], lookback: int) -> pd.DataFrame:
        nonlocal runs
        runs += 1
        return _daily_frame(base)

    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=1)
    supervisor = TaskSupervisor(namespace="operational-backbone-service-test")

    try:
        await service.start_scheduler(
            _plan,
            schedule,
            fetch_daily=_fetch_daily,
            task_supervisor=supervisor,
        )
        await asyncio.sleep(0.2)
        assert runs >= 1
        assert service.scheduler_running is True
    finally:
        await service.stop_scheduler()
        await service.shutdown()
        await event_bus.stop()
        flush = getattr(cache.raw_client, "flushall", None)
        if callable(flush):  # pragma: no branch - optional dependency cleanup
            flush()

    assert service.scheduler_running is False
