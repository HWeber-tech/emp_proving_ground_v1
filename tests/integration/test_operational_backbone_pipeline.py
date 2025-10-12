from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pandas as pd
import pytest

from src.core.event_bus import Event, EventBus
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.in_memory_broker import InMemoryKafkaBroker
from src.data_foundation.streaming.kafka_stream import (
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
)
from src.data_integration.real_data_integration import RealDataManager
from src.runtime.task_supervisor import TaskSupervisor
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.thinking.adaptation.policy_router import PolicyRouter, PolicyTactic
from src.understanding.router import UnderstandingRouter


KAFKA_TOPICS = {
    "daily_bars": "telemetry.ingest",
    "intraday_trades": "telemetry.ingest",
    "macro_events": "telemetry.ingest",
}


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


def _macro_events(base: datetime) -> list[dict[str, object]]:
    return [
        {
            "timestamp": base - timedelta(days=1, hours=2),
            "calendar": "ECB",
            "event": "Rate Decision",
            "currency": "EUR",
            "actual": 4.0,
            "forecast": 4.0,
            "previous": 3.5,
            "importance": "high",
        },
        {
            "timestamp": base,
            "calendar": "FED",
            "event": "FOMC Minutes",
            "currency": "USD",
            "actual": None,
            "forecast": None,
            "previous": None,
            "importance": "medium",
        },
    ]


def _build_manager(
    settings: TimescaleConnectionSettings,
    broker: InMemoryKafkaBroker,
    topic_map: dict[str, str],
) -> tuple[RealDataManager, ManagedRedisCache]:
    cache = ManagedRedisCache(InMemoryRedis(), RedisCachePolicy.institutional_defaults())
    publisher = KafkaIngestEventPublisher(
        broker.create_producer(),
        topic_map=topic_map,
    )
    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
        managed_cache=cache,
    )
    return manager, cache


def _consumer_factory(
    broker: InMemoryKafkaBroker,
    event_bus: EventBus,
    *,
    topics: tuple[str, ...] = ("telemetry.ingest",),
    poll_timeout: float = 0.05,
    idle_sleep: float = 0.0,
) -> KafkaIngestEventConsumer:
    consumer = broker.create_consumer()
    return KafkaIngestEventConsumer(
        consumer,
        topics=topics,
        event_bus=event_bus,
        poll_timeout=poll_timeout,
        idle_sleep=idle_sleep,
        publish_consumer_lag=False,
    )


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_full_cycle(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'pipeline_timescale.db'}")
    broker = InMemoryKafkaBroker()
    manager, cache = _build_manager(settings, broker, KAFKA_TOPICS)
    event_bus = EventBus()
    sensory = RealSensoryOrgan()

    def consumer_factory() -> KafkaIngestEventConsumer:
        return _consumer_factory(broker, event_bus)

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
        sensory_organ=sensory,
        event_topics=("telemetry.ingest",),
    )

    base = datetime(2024, 5, 6, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
        macro_events=_macro_events(base),
    )

    try:
        result = await pipeline.execute(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
        )
    finally:
        await pipeline.shutdown()

    messages = broker.snapshot()
    assert messages, "Kafka publisher should record ingest telemetry"
    assert {message.topic for message in messages} == {"telemetry.ingest"}
    assert len(messages) == len(result.kafka_events)

    assert "daily_bars" in result.ingest_results
    assert result.ingest_results["daily_bars"].rows_written == 2
    assert "intraday_trades" in result.ingest_results
    assert "macro_events" in result.ingest_results

    daily_frame = result.frames["daily_bars"]
    assert daily_frame.iloc[-1]["close"] == pytest.approx(1.125)
    intraday_frame = result.frames["intraday_trades"]
    assert intraday_frame.iloc[-1]["price"] == pytest.approx(1.124)
    assert not result.frames["macro_events"].empty

    cache_metrics = cache.metrics(reset=False)
    assert int(cache_metrics.get("misses", 0)) >= 1

    snapshot = result.sensory_snapshot
    assert snapshot is not None
    assert snapshot["symbol"] == "EURUSD"
    integrated = snapshot["integrated_signal"]
    assert float(getattr(integrated, "confidence")) >= 0.0

    assert result.ingest_error is None
    assert not event_bus.is_running()


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_understanding_failover(
    tmp_path, monkeypatch
) -> None:
    settings = TimescaleConnectionSettings(
        url=f"sqlite:///{tmp_path / 'pipeline_timescale_understanding.db'}"
    )
    broker = InMemoryKafkaBroker()
    manager, cache = _build_manager(settings, broker, KAFKA_TOPICS)
    event_bus = EventBus()
    sensory = RealSensoryOrgan()

    policy_router = PolicyRouter()
    policy_router.register_tactic(
        PolicyTactic(
            tactic_id="momentum",
            base_weight=1.0,
            parameters={"style": "momentum"},
            guardrails={"risk_cap": "shadow"},
        )
    )
    understanding_router = UnderstandingRouter(policy_router)

    def consumer_factory() -> KafkaIngestEventConsumer:
        return _consumer_factory(broker, event_bus)

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
        sensory_organ=sensory,
        understanding_router=understanding_router,
        event_topics=("telemetry.ingest",),
    )

    base = datetime(2024, 5, 6, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
    )

    try:
        result = await pipeline.execute(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
            poll_consumer=False,
        )

        assert result.ingest_error is None
        assert result.belief_state is not None
        assert result.regime_signal is not None
        assert result.understanding_decision is not None

        monkeypatch.setattr(
            manager,
            "ingest_market_slice",
            lambda *_, **__: (_ for _ in ()).throw(RuntimeError("Timescale offline")),
        )
        monkeypatch.setattr(
            manager._reader,
            "fetch_daily_bars",
            lambda *_, **__: (_ for _ in ()).throw(RuntimeError("Timescale unreachable")),
        )
        monkeypatch.setattr(
            manager._reader,
            "fetch_intraday_trades",
            lambda *_, **__: (_ for _ in ()).throw(RuntimeError("Timescale unreachable")),
        )

        result_failover = await pipeline.execute(
            request,
            poll_consumer=False,
        )

        assert result_failover.ingest_error is not None
        assert "daily_bars" in result_failover.frames
        assert not result_failover.frames["daily_bars"].empty
        assert result_failover.understanding_decision is not None
    finally:
        await pipeline.shutdown()
        cache.metrics(reset=True)


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_streaming_supervision(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'pipeline_streaming.db'}")
    broker = InMemoryKafkaBroker()
    manager, cache = _build_manager(settings, broker, {"daily_bars": "telemetry.ingest"})
    event_bus = EventBus()
    sensory = RealSensoryOrgan()
    supervisor = TaskSupervisor(namespace="operational-stream-test")

    await event_bus.start()

    received: list[Event] = []
    received_event = asyncio.Event()

    def _collect(event: Event) -> None:
        received.append(event)
        received_event.set()

    subscription = event_bus.subscribe("telemetry.ingest", _collect)

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=lambda: _consumer_factory(
            broker,
            event_bus,
            poll_timeout=0.01,
            idle_sleep=0.01,
        ),
        sensory_organ=sensory,
        auto_close_consumer=False,
        task_supervisor=supervisor,
        event_topics=("telemetry.ingest",),
    )

    streaming_task = await pipeline.start_streaming()
    assert streaming_task is not None
    assert supervisor.active_count == 1

    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=None,
    )

    await pipeline.execute(
        request,
        fetch_daily=lambda symbols, lookback: _daily_frame(base),
        fetch_intraday=lambda symbols, lookback, interval: pd.DataFrame(),
        poll_consumer=False,
    )

    await asyncio.wait_for(received_event.wait(), timeout=1.0)
    assert received
    assert received[-1].type == "telemetry.ingest"

    await pipeline.stop_streaming()
    assert supervisor.active_count == 0

    await pipeline.shutdown()
    await event_bus.stop()
    event_bus.unsubscribe(subscription)
    cache.metrics(reset=True)


@pytest.mark.asyncio()
async def test_operational_backbone_streaming_feeds_sensory(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'pipeline_streaming_sensory.db'}")
    broker = InMemoryKafkaBroker()
    manager, cache = _build_manager(settings, broker, KAFKA_TOPICS)
    event_bus = EventBus()
    sensory = RealSensoryOrgan(event_bus=event_bus)
    supervisor = TaskSupervisor(namespace="operational-stream-sensory-test")

    snapshots: list[Mapping[str, Any]] = []

    def _collect_snapshot(snapshot: Mapping[str, Any]) -> None:
        snapshots.append(snapshot)

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=lambda: _consumer_factory(
            broker,
            event_bus,
            poll_timeout=0.01,
            idle_sleep=0.01,
        ),
        sensory_organ=sensory,
        task_supervisor=supervisor,
        sensory_snapshot_callback=_collect_snapshot,
    )

    streaming_task = await pipeline.start_streaming()
    assert streaming_task is not None

    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
    )

    await pipeline.execute(
        request,
        fetch_daily=lambda symbols, lookback: _daily_frame(base),
        fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
        poll_consumer=False,
    )

    async def _wait_for_snapshot() -> None:
        for _ in range(40):
            if snapshots:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("expected streaming sensory snapshot")

    await _wait_for_snapshot()

    latest_snapshot = pipeline.streaming_snapshots.get("EURUSD")
    assert latest_snapshot is not None
    assert latest_snapshot["symbol"] == "EURUSD"

    await pipeline.shutdown()
    cache.metrics(reset=True)
