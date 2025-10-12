from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.in_memory_broker import InMemoryKafkaBroker
from src.data_foundation.streaming.kafka_stream import (
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
)
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.real_sensory_organ import RealSensoryOrgan


try:  # pragma: no cover - optional dependency mirrors production Redis wiring
    import fakeredis
except Exception:  # pragma: no cover
    fakeredis = None  # type: ignore[assignment]


def _managed_cache(policy: RedisCachePolicy | None = None) -> ManagedRedisCache:
    policy = policy or RedisCachePolicy.institutional_defaults()
    if fakeredis is not None:
        client = fakeredis.FakeRedis()
    else:
        client = InMemoryRedis()
    return ManagedRedisCache(client, policy)


def _flush_cache(cache: ManagedRedisCache) -> None:
    flush = getattr(cache.raw_client, "flushall", None)
    if callable(flush):  # pragma: no branch - optional dependency cleanup
        flush()


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


def _build_manager(
    settings: TimescaleConnectionSettings,
    broker: InMemoryKafkaBroker,
) -> tuple[RealDataManager, ManagedRedisCache, KafkaIngestEventPublisher]:
    cache = _managed_cache(RedisCachePolicy.institutional_defaults())
    publisher = KafkaIngestEventPublisher(
        broker.create_producer(),
        topic_map={"daily_bars": "telemetry.ingest", "intraday_trades": "telemetry.ingest"},
    )
    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
        managed_cache=cache,
    )
    return manager, cache, publisher


@pytest.mark.asyncio()
async def test_operational_backbone_streams_into_sensory(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'operational_ts.db'}")
    broker = InMemoryKafkaBroker()
    manager, cache, _publisher = _build_manager(settings, broker)

    try:
        base = datetime(2024, 3, 5, tzinfo=timezone.utc)
        plan = TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD",), lookback_days=2),
            intraday=IntradayTradeIngestPlan(symbols=("EURUSD",), lookback_days=1, interval="1m"),
        )

        manager.run_ingest_plan(
            plan,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
        )

        messages = broker.snapshot()
        assert messages, "Kafka publisher should emit ingest telemetry"
        decoded = json.loads(messages[0].value.decode("utf-8"))
        assert decoded["result"]["dimension"] in {"daily_bars", "intraday_trades"}

        daily_frame = manager.fetch_data("EURUSD", interval="1d", period="2d")
        assert not daily_frame.empty
        assert daily_frame.iloc[-1]["close"] == pytest.approx(1.125)

        intraday_frame = manager.fetch_data("EURUSD", interval="1m")
        assert not intraday_frame.empty
        assert intraday_frame.iloc[-1]["price"] == pytest.approx(1.124)

        metrics = cache.metrics(reset=True)
        assert int(metrics.get("misses", 0)) >= 1

        events: list[Any] = []
        event_bus = EventBus()
        await event_bus.start()
        try:
            event_bus.subscribe("telemetry.ingest", lambda event: events.append(event))

            bridge = KafkaIngestEventConsumer(
                broker.create_consumer(),
                topics=("telemetry.ingest",),
                event_bus=event_bus,
                publish_consumer_lag=False,
            )
            try:
                processed = False
                while bridge.poll_once():
                    processed = True
                assert processed, "Kafka bridge should process at least one message"
                await asyncio.sleep(0.05)
            finally:
                bridge.close()
        finally:
            await event_bus.stop()

        assert events, "Event bus should receive telemetry from Kafka bridge"
        first_payload = events[0].payload
        assert first_payload["result"]["dimension"] in {"daily_bars", "intraday_trades"}

        organ = RealSensoryOrgan()
        snapshot = organ.observe(daily_frame, symbol="EURUSD")
        assert snapshot["symbol"] == "EURUSD"
        assert {"WHAT", "ANOMALY"}.issubset(snapshot["dimensions"].keys())
    finally:
        await manager.shutdown()
        cache.metrics(reset=True)
        _flush_cache(cache)
