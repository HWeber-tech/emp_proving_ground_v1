from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd
import pytest

from src.core.event_bus import EventBus, Event
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.kafka_stream import (
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
)
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.thinking.adaptation.policy_router import PolicyRouter, PolicyTactic
from src.understanding.router import UnderstandingRouter
from src.runtime.task_supervisor import TaskSupervisor


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


class _StreamingKafkaConsumer:
    def __init__(self, producer: _FakeKafkaProducer) -> None:
        self._producer = producer
        self._cursor = 0
        self._subscribed: tuple[str, ...] = ()
        self.closed = False

    def subscribe(self, topics: Iterable[str]) -> None:
        self._subscribed = tuple(str(topic) for topic in topics)

    def poll(self, timeout: float | None = None) -> _FakeKafkaMessage | None:
        if self._cursor >= len(self._producer.messages):
            return None
        payload = dict(self._producer.messages[self._cursor])
        self._cursor += 1
        return _FakeKafkaMessage(payload)

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


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_full_cycle(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'pipeline_timescale.db'}"
    settings = TimescaleConnectionSettings(url=url)

    producer = _FakeKafkaProducer()
    publisher = KafkaIngestEventPublisher(
        producer,
        topic_map={
            "daily_bars": "telemetry.ingest",
            "intraday_trades": "telemetry.ingest",
        },
    )

    manager = RealDataManager(timescale_settings=settings, ingest_publisher=publisher)
    event_bus = EventBus()
    sensory = RealSensoryOrgan()

    base = datetime(2024, 5, 6, tzinfo=timezone.utc)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1m",
        macro_events=_macro_events(base),
    )

    def consumer_factory() -> KafkaIngestEventConsumer:
        consumer = _FakeKafkaConsumer(dict(message) for message in producer.messages)
        return KafkaIngestEventConsumer(
            consumer,
            topics=("telemetry.ingest",),
            event_bus=event_bus,
            poll_timeout=0.1,
            idle_sleep=0.0,
        )

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
        sensory_organ=sensory,
        event_topics=("telemetry.ingest",),
    )

    try:
        result = await pipeline.execute(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
        )
    finally:
        await pipeline.shutdown()

    assert "daily_bars" in result.ingest_results
    assert result.ingest_results["daily_bars"].rows_written == 2
    assert "intraday_trades" in result.ingest_results
    assert "macro_events" in result.ingest_results
    assert result.frames["daily_bars"].iloc[-1]["close"] == pytest.approx(1.125)
    assert result.frames["intraday_trades"].iloc[-1]["price"] == pytest.approx(1.124)
    assert not result.frames["macro_events"].empty
    macro_frame = result.frames["macro_events"]
    assert set(macro_frame["calendar"].unique()) == {"ECB", "FED"}

    assert len(producer.messages) == len(result.kafka_events) > 0
    assert all(event.type == "telemetry.ingest" for event in result.kafka_events)

    hits = int(result.cache_metrics_after_fetch.get("hits", 0))
    assert hits >= 1

    assert result.sensory_snapshot is not None
    assert result.sensory_snapshot["symbol"] == "EURUSD"
    integrated = result.sensory_snapshot["integrated_signal"]
    assert float(getattr(integrated, "confidence")) >= 0.0
    assert result.ingest_error is None
    assert result.belief_state is None
    assert result.understanding_decision is None

    assert not event_bus.is_running()


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_understanding_failover(
    tmp_path, monkeypatch
) -> None:
    url = f"sqlite:///{tmp_path / 'pipeline_timescale_understanding.db'}"
    settings = TimescaleConnectionSettings(url=url)

    producer = _FakeKafkaProducer()
    publisher = KafkaIngestEventPublisher(
        producer,
        topic_map={
            "daily_bars": "telemetry.ingest",
            "intraday_trades": "telemetry.ingest",
        },
    )

    manager = RealDataManager(timescale_settings=settings, ingest_publisher=publisher)
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
        consumer = _FakeKafkaConsumer(dict(message) for message in producer.messages)
        return KafkaIngestEventConsumer(
            consumer,
            topics=("telemetry.ingest",),
            event_bus=event_bus,
            poll_timeout=0.1,
            idle_sleep=0.0,
        )

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
        sensory_organ=sensory,
        understanding_router=understanding_router,
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


@pytest.mark.asyncio()
async def test_operational_backbone_pipeline_streaming_supervision(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'pipeline_streaming.db'}"
    settings = TimescaleConnectionSettings(url=url)

    producer = _FakeKafkaProducer()
    publisher = KafkaIngestEventPublisher(
        producer,
        topic_map={
            "daily_bars": "telemetry.ingest",
        },
    )

    manager = RealDataManager(timescale_settings=settings, ingest_publisher=publisher)
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

    streaming_consumer = _StreamingKafkaConsumer(producer)

    def consumer_factory() -> KafkaIngestEventConsumer:
        return KafkaIngestEventConsumer(
            streaming_consumer,
            topics=("telemetry.ingest",),
            event_bus=event_bus,
            poll_timeout=0.01,
            idle_sleep=0.01,
        )

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
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
