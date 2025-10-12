"""Operational data backbone pipeline orchestrating ingest, cache, and streaming."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

import pandas as pd

from src.core.event_bus import Event, EventBus, SubscriptionHandle
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    create_ingest_event_consumer,
    ingest_topic_config_from_mapping,
)
from src.data_integration.real_data_integration import RealDataManager
from src.governance.system_config import SystemConfig
from src.sensory.real_sensory_organ import RealSensoryOrgan

if TYPE_CHECKING:  # pragma: no cover - typing support only
    from src.runtime.task_supervisor import TaskSupervisor

FetchDailyFn = Callable[[list[str], int], pd.DataFrame]
FetchIntradayFn = Callable[[list[str], int, str], pd.DataFrame]
FetchMacroFn = Callable[[str, str], Sequence[Mapping[str, object]]]
KafkaConsumerFactory = Callable[[], KafkaIngestEventConsumer | None]


def _normalise_symbols(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        token = str(value).strip().upper()
        if token and token not in seen:
            seen.append(token)
    if not seen:
        raise ValueError("symbols must contain at least one non-empty entry")
    return tuple(seen)


@dataclass(slots=True, frozen=True)
class OperationalIngestRequest:
    """Describe the ingestion slice for the operational backbone."""

    symbols: Sequence[str]
    daily_lookback_days: int | None = 60
    intraday_lookback_days: int | None = 2
    intraday_interval: str = "1m"
    macro_start: str | None = None
    macro_end: str | None = None
    macro_events: Sequence[Mapping[str, object]] | None = None
    source: str = "yahoo"
    macro_source: str = "fred"

    def normalised_symbols(self) -> tuple[str, ...]:
        return _normalise_symbols(self.symbols)

    def primary_symbol(self) -> str:
        return self.normalised_symbols()[0]

    def daily_period(self) -> str | None:
        if self.daily_lookback_days is None:
            return None
        return f"{max(int(self.daily_lookback_days), 1)}d"

    def intraday_period(self) -> str | None:
        if self.intraday_lookback_days is None:
            return None
        return f"{max(int(self.intraday_lookback_days), 1)}d"


@dataclass(slots=True, frozen=True)
class OperationalBackboneResult:
    """Structured output produced by :class:`OperationalBackbonePipeline`."""

    ingest_results: Mapping[str, TimescaleIngestResult]
    frames: Mapping[str, pd.DataFrame]
    kafka_events: tuple[Event, ...]
    cache_metrics_before: Mapping[str, int | float | str]
    cache_metrics_after_ingest: Mapping[str, int | float | str]
    cache_metrics_after_fetch: Mapping[str, int | float | str]
    sensory_snapshot: Mapping[str, Any] | None = None


class OperationalBackbonePipeline:
    """Coordinate Timescale ingest, Redis cache warming, and Kafka bridging."""

    def __init__(
        self,
        *,
        manager: RealDataManager,
        event_bus: EventBus | None = None,
        kafka_consumer: KafkaIngestEventConsumer | None = None,
        kafka_consumer_factory: KafkaConsumerFactory | None = None,
        sensory_organ: RealSensoryOrgan | None = None,
        event_topics: Sequence[str] | None = None,
        auto_close_consumer: bool = True,
    ) -> None:
        if kafka_consumer is not None and kafka_consumer_factory is not None:
            raise ValueError("Provide either kafka_consumer or kafka_consumer_factory, not both")
        self._manager = manager
        self._event_bus = event_bus
        self._consumer = kafka_consumer
        self._consumer_factory = kafka_consumer_factory
        self._sensory_organ = sensory_organ
        self._event_topics = tuple(event_topics or ("telemetry.ingest",))
        self._auto_close_consumer = auto_close_consumer
        self._manager_shutdown = False

    async def execute(
        self,
        request: OperationalIngestRequest,
        *,
        fetch_daily: FetchDailyFn | None = None,
        fetch_intraday: FetchIntradayFn | None = None,
        fetch_macro: FetchMacroFn | None = None,
        poll_consumer: bool = True,
    ) -> OperationalBackboneResult:
        events: list[Event] = []
        subscriptions: list[SubscriptionHandle] = []
        bus_started_here = False

        if self._event_bus is not None:
            if not self._event_bus.is_running():
                await self._event_bus.start()
                bus_started_here = True

            def _collector(event: Event) -> None:
                events.append(event)

            for topic in self._event_topics:
                subscriptions.append(self._event_bus.subscribe(topic, _collector))

        try:
            cache_before = self._manager.cache_metrics(reset=True)
            ingest_results = await asyncio.to_thread(
                self._manager.ingest_market_slice,
                symbols=request.symbols,
                daily_lookback_days=request.daily_lookback_days,
                intraday_lookback_days=request.intraday_lookback_days,
                intraday_interval=request.intraday_interval,
                macro_start=request.macro_start,
                macro_end=request.macro_end,
                macro_events=request.macro_events,
                source=request.source,
                macro_source=request.macro_source,
                fetch_daily=fetch_daily,
                fetch_intraday=fetch_intraday,
                fetch_macro=fetch_macro,
            )
            cache_after_ingest = self._manager.cache_metrics(reset=False)

            frames: MutableMapping[str, pd.DataFrame] = {}
            symbol = request.primary_symbol()

            async def _fetch_and_warm(
                *,
                dimension: str,
                interval: str,
                period: str | None,
            ) -> pd.DataFrame:
                frame = await asyncio.to_thread(
                    self._manager.fetch_data,
                    symbol,
                    period=period,
                    interval=interval,
                )
                if not frame.empty:
                    await asyncio.to_thread(
                        self._manager.fetch_data,
                        symbol,
                        period=period,
                        interval=interval,
                    )
                frames[dimension] = frame
                return frame

            if "daily_bars" in ingest_results:
                await _fetch_and_warm(
                    dimension="daily_bars",
                    interval="1d",
                    period=request.daily_period(),
                )

            if "intraday_trades" in ingest_results:
                await _fetch_and_warm(
                    dimension="intraday_trades",
                    interval=request.intraday_interval,
                    period=request.intraday_period(),
                )

            cache_after_fetch = self._manager.cache_metrics(reset=False)

            sensory_snapshot: Mapping[str, Any] | None = None
            if (
                self._sensory_organ is not None
                and "daily_bars" in frames
                and not frames["daily_bars"].empty
            ):
                sensory_snapshot = await asyncio.to_thread(
                    self._sensory_organ.observe,
                    frames["daily_bars"],
                    symbol=symbol,
                )

            consumer, owns_consumer = self._resolve_consumer()
            if poll_consumer and consumer is not None:
                while True:
                    processed = await asyncio.to_thread(consumer.poll_once)
                    if not processed:
                        break
                if owns_consumer or self._auto_close_consumer:
                    consumer.close()

            return OperationalBackboneResult(
                ingest_results=dict(ingest_results),
                frames=dict(frames),
                kafka_events=tuple(events),
                cache_metrics_before=dict(cache_before),
                cache_metrics_after_ingest=dict(cache_after_ingest),
                cache_metrics_after_fetch=dict(cache_after_fetch),
                sensory_snapshot=sensory_snapshot,
            )
        finally:
            if self._event_bus is not None:
                for handle in subscriptions:
                    self._event_bus.unsubscribe(handle)
                if bus_started_here:
                    await self._event_bus.stop()

    async def shutdown(self) -> None:
        if self._manager_shutdown:
            return
        await self._manager.shutdown()
        self._manager_shutdown = True

    def _resolve_consumer(self) -> tuple[KafkaIngestEventConsumer | None, bool]:
        if self._consumer_factory is not None:
            consumer = self._consumer_factory()
            return consumer, True if consumer is not None else False
        if self._consumer is not None:
            return self._consumer, False
        return None, False


def create_operational_backbone_pipeline(
    system_config: SystemConfig,
    *,
    event_bus: EventBus,
    sensory_organ: RealSensoryOrgan | None = None,
    task_supervisor: "TaskSupervisor | None" = None,
    extras_override: Mapping[str, str] | None = None,
    kafka_consumer: KafkaIngestEventConsumer | None = None,
    kafka_consumer_factory: KafkaConsumerFactory | None = None,
    kafka_deserializer: Callable[[bytes | bytearray | str], Mapping[str, object]] | None = None,
    event_topics: Sequence[str] | None = None,
    auto_close_consumer: bool | None = None,
    manager_kwargs: Mapping[str, object] | None = None,
) -> OperationalBackbonePipeline:
    """Instantiate an operational backbone pipeline from ``SystemConfig``.

    The helper wires together a :class:`RealDataManager` with the configured
    Timescale/Redis/Kafka settings, creates an optional Kafka ingest bridge, and
    returns a ready-to-run :class:`OperationalBackbonePipeline`.  Callers can
    provide overrides for specific manager settings (e.g. stub publishers in
    tests) via ``manager_kwargs``.
    """

    extras: dict[str, str] = dict(system_config.extras or {})
    if extras_override:
        extras.update({str(key): str(value) for key, value in extras_override.items()})

    manager_params = dict(manager_kwargs or {})
    manager_params.setdefault("system_config", system_config)
    manager_params.setdefault("extras", extras)
    if task_supervisor is not None:
        manager_params.setdefault("task_supervisor", task_supervisor)

    manager = RealDataManager(**manager_params)

    created_consumer = False
    bridge = kafka_consumer
    if bridge is None:
        kafka_settings = KafkaConnectionSettings.from_mapping(extras)
        bridge = create_ingest_event_consumer(
            kafka_settings,
            extras,
            event_bus=event_bus,
            consumer_factory=kafka_consumer_factory,
            deserializer=kafka_deserializer,
        )
        created_consumer = bridge is not None

    if event_topics is not None:
        resolved_topics = tuple(
            dict.fromkeys(str(topic).strip() for topic in event_topics if str(topic).strip())
        )
    else:
        topic_map, default_topic = ingest_topic_config_from_mapping(extras)
        candidate_topics = [
            str(default_topic).strip() if default_topic else "telemetry.ingest"
        ]
        candidate_topics.extend(topic_map.values())
        if sensory_organ is not None:
            candidate_topics.append("telemetry.sensory.snapshot")
        resolved_topics = tuple(
            dict.fromkeys(topic for topic in candidate_topics if topic)
        )

    if auto_close_consumer is None:
        auto_close_consumer = created_consumer

    try:
        pipeline = OperationalBackbonePipeline(
            manager=manager,
            event_bus=event_bus,
            kafka_consumer=bridge,
            sensory_organ=sensory_organ,
            event_topics=resolved_topics,
            auto_close_consumer=bool(auto_close_consumer),
        )
    except Exception:
        manager.close()
        if created_consumer and bridge is not None:
            bridge.close()
        raise

    return pipeline


__all__ = [
    "OperationalBackbonePipeline",
    "OperationalBackboneResult",
    "OperationalIngestRequest",
    "create_operational_backbone_pipeline",
]
