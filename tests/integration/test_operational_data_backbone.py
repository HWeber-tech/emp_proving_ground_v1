from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.streaming.kafka_stream import (
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
)
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.real_sensory_organ import RealSensoryOrgan


class _FakeKafkaProducer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def produce(self, topic: str, value: bytes, key: str | bytes | None = None) -> None:
        self.messages.append({"topic": topic, "value": value, "key": key})

    def flush(self, timeout: float | None = None) -> None:  # pragma: no cover - noop
        return None


@dataclass
class _FakeKafkaMessage:
    payload: dict[str, Any]

    def error(self) -> None:
        return None

    def value(self) -> bytes:
        return self.payload["value"]

    def topic(self) -> str:
        return self.payload["topic"]

    def key(self) -> str | bytes | None:
        return self.payload.get("key")


class _FakeKafkaConsumer:
    def __init__(self, messages: Iterable[dict[str, Any]]) -> None:
        self._queue = deque(messages)
        self._subscribed: tuple[str, ...] = ()
        self.closed = False

    def subscribe(self, topics: Iterable[str]) -> None:
        self._subscribed = tuple(str(topic) for topic in topics)

    def poll(self, timeout: float | None = None) -> _FakeKafkaMessage | None:
        if not self._queue:
            return None
        return _FakeKafkaMessage(self._queue.popleft())

    def close(self) -> None:
        self.closed = True


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


@pytest.mark.asyncio()
async def test_operational_backbone_streams_into_sensory(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'operational_timescale.db'}"
    settings = TimescaleConnectionSettings(url=url)

    producer = _FakeKafkaProducer()
    publisher = KafkaIngestEventPublisher(
        producer,
        topic_map={
            "daily_bars": "telemetry.ingest",
            "intraday_trades": "telemetry.ingest",
        },
    )

    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
    )

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

        assert producer.messages, "Kafka publisher should emit ingest events"
        decoded = json.loads(producer.messages[0]["value"].decode("utf-8"))
        assert decoded["result"]["dimension"] in {"daily_bars", "intraday_trades"}

        daily_frame = manager.fetch_data("EURUSD", interval="1d", period="2d")
        assert not daily_frame.empty
        assert daily_frame.iloc[-1]["close"] == pytest.approx(1.125)

        intraday_frame = manager.fetch_data("EURUSD", interval="1m")
        assert not intraday_frame.empty
        assert intraday_frame.iloc[-1]["price"] == pytest.approx(1.124)

        metrics = manager.cache_metrics(reset=True)
        assert metrics["misses"] >= 1

        events: list[Any] = []
        event_bus = EventBus()
        await event_bus.start()
        try:
            event_bus.subscribe("telemetry.ingest", lambda event: events.append(event))

            consumer = _FakeKafkaConsumer(dict(message) for message in producer.messages)
            bridge = KafkaIngestEventConsumer(
                consumer,
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
        dimensions = snapshot["dimensions"]
        assert {"WHAT", "ANOMALY"}.issubset(dimensions.keys())
        integrated = snapshot["integrated_signal"]
        assert float(getattr(integrated, "confidence")) >= 0.0
    finally:
        await manager.shutdown()
