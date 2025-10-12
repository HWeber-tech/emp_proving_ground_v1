"""Operational data backbone pipeline orchestrating ingest, cache, and streaming."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import pandas as pd

from src.core.event_bus import Event, EventBus, SubscriptionHandle
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    create_ingest_event_consumer,
    ingest_topic_config_from_mapping,
)
from src.data_foundation.ingest.telemetry import CompositeIngestPublisher
from src.data_integration.real_data_integration import RealDataManager
from src.governance.system_config import SystemConfig
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.understanding.belief import BeliefState, RegimeSignal
from src.understanding.live_belief_manager import LiveBeliefManager
from src.understanding.router import BeliefSnapshot, UnderstandingDecision, UnderstandingRouter
from src.runtime.task_supervisor import TaskSupervisor

FetchDailyFn = Callable[[list[str], int], pd.DataFrame]
FetchIntradayFn = Callable[[list[str], int, str], pd.DataFrame]
FetchMacroFn = Callable[[str, str], Sequence[Mapping[str, object]]]
KafkaConsumerFactory = Callable[[], KafkaIngestEventConsumer | None]


logger = logging.getLogger(__name__)


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

    def macro_calendars(self) -> tuple[str, ...]:
        if not self.macro_events:
            return tuple()
        calendars: list[str] = []
        for entry in self.macro_events:
            candidate: object | None = None
            if isinstance(entry, Mapping):
                candidate = entry.get("calendar") or entry.get("CALENDAR")
            else:
                candidate = getattr(entry, "calendar", None)
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text and text not in calendars:
                calendars.append(text)
        return tuple(calendars)


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
    belief_state: BeliefState | None = None
    regime_signal: RegimeSignal | None = None
    belief_snapshot: BeliefSnapshot | None = None
    understanding_decision: UnderstandingDecision | None = None
    ingest_error: str | None = None


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
        belief_manager: LiveBeliefManager | None = None,
        understanding_router: UnderstandingRouter | None = None,
        belief_id: str | None = None,
        task_supervisor: TaskSupervisor | None = None,
        streaming_task_name: str | None = None,
        streaming_metadata: Mapping[str, object] | None = None,
        stream_sensory_from_kafka: bool = True,
        sensory_snapshot_callback: Callable[[Mapping[str, Any]], None] | None = None,
        shutdown_manager_on_close: bool = True,
    ) -> None:
        if kafka_consumer is not None and kafka_consumer_factory is not None:
            raise ValueError("Provide either kafka_consumer or kafka_consumer_factory, not both")
        if understanding_router is not None and event_bus is None and belief_manager is None:
            raise ValueError(
                "event_bus must be provided when enabling understanding routing without a pre-built belief manager",
            )
        self._manager = manager
        self._event_bus = event_bus
        self._consumer = kafka_consumer
        self._consumer_factory = kafka_consumer_factory
        self._sensory_organ = sensory_organ
        self._event_topics = tuple(event_topics or ("telemetry.ingest",))
        self._auto_close_consumer = auto_close_consumer
        self._manager_shutdown = False
        self._belief_manager = belief_manager
        self._understanding_router = understanding_router
        self._belief_id = belief_id
        self._task_supervisor = task_supervisor
        self._owns_task_supervisor = False
        self._streaming_task: asyncio.Task[None] | None = None
        self._streaming_stop_event: asyncio.Event | None = None
        self._streaming_consumer: KafkaIngestEventConsumer | None = None
        self._streaming_consumer_owned = False
        self._streaming_task_name = streaming_task_name or "operational.kafka.stream"
        self._streaming_metadata = dict(streaming_metadata) if streaming_metadata else None
        self._streaming_started_bus = False
        self._stream_sensory_from_kafka = bool(stream_sensory_from_kafka)
        self._sensory_snapshot_callback = sensory_snapshot_callback
        self._streaming_subscriptions: list[SubscriptionHandle] | None = None
        self._streaming_snapshots: dict[str, Mapping[str, Any]] = {}
        self._shutdown_manager_on_close = bool(shutdown_manager_on_close)

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
            ingest_error: str | None = None
            try:
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
            except Exception as exc:
                ingest_results = {}
                ingest_error = str(exc)
                logger.warning(
                    "Operational backbone ingest failed; continuing with cached data: %s",
                    exc,
                    exc_info=True,
                )
            cache_after_ingest = self._manager.cache_metrics(reset=False)

            frames: MutableMapping[str, pd.DataFrame] = {}
            symbol = request.primary_symbol()

            async def _fetch_and_warm(
                *,
                dimension: str,
                interval: str | None,
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
            elif (
                ingest_error is not None
                and request.daily_lookback_days is not None
                and request.daily_lookback_days > 0
            ):
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
            elif (
                ingest_error is not None
                and request.intraday_lookback_days is not None
                and request.intraday_lookback_days > 0
            ):
                await _fetch_and_warm(
                    dimension="intraday_trades",
                    interval=request.intraday_interval,
                    period=request.intraday_period(),
                )

            macro_result = ingest_results.get("macro_events")
            macro_requested = (
                "macro_events" in ingest_results
                or request.macro_events is not None
                or (request.macro_start is not None and request.macro_end is not None)
            )
            macro_start = request.macro_start or (
                macro_result.start_ts if isinstance(macro_result, TimescaleIngestResult) else None
            )
            macro_end = request.macro_end or (
                macro_result.end_ts if isinstance(macro_result, TimescaleIngestResult) else None
            )
            macro_calendars = request.macro_calendars()

            async def _fetch_macro_events() -> pd.DataFrame:
                frame = await asyncio.to_thread(
                    self._manager.fetch_macro_events,
                    calendars=macro_calendars or None,
                    start=macro_start,
                    end=macro_end,
                )
                frames["macro_events"] = frame
                return frame

            if "macro_events" in ingest_results:
                await _fetch_macro_events()
            elif ingest_error is not None and macro_requested:
                await _fetch_macro_events()

            cache_after_fetch = self._manager.cache_metrics(reset=False)

            observed_events: dict[str, Event] = {}
            for event in events:
                payload = event.payload if isinstance(event.payload, Mapping) else None
                if not isinstance(payload, Mapping):
                    continue
                result_blob = payload.get("result")
                if isinstance(result_blob, Mapping):
                    dimension_token = result_blob.get("dimension")
                    if isinstance(dimension_token, str) and dimension_token not in observed_events:
                        observed_events[dimension_token] = event

            primary_topic = self._event_topics[0] if self._event_topics else "telemetry.ingest"

            expected_dimensions: set[str] = set()

            def _collect_expected_dimensions(publisher: object) -> None:
                if publisher is None:
                    return
                if isinstance(publisher, CompositeIngestPublisher):
                    for child in getattr(publisher, "_publishers", ()):  # type: ignore[attr-defined]
                        _collect_expected_dimensions(child)
                    return
                topic_map = getattr(publisher, "_topic_map", None)
                default_topic = getattr(publisher, "_default_topic", None)
                if isinstance(topic_map, Mapping):
                    for dimension_key in topic_map.keys():
                        if dimension_key == "*":
                            expected_dimensions.update(ingest_results.keys())
                        else:
                            expected_dimensions.add(str(dimension_key))
                if default_topic:
                    expected_dimensions.update(ingest_results.keys())

            kafka_publisher = getattr(self._manager, "_kafka_publisher", None)
            _collect_expected_dimensions(kafka_publisher)
            if not expected_dimensions:
                expected_dimensions = set(ingest_results.keys())

            for dimension, ingest_result in ingest_results.items():
                if dimension not in expected_dimensions:
                    continue
                if dimension in observed_events:
                    continue
                if ingest_result.rows_written <= 0:
                    continue
                events.append(
                    Event(
                        type=primary_topic,
                        payload={
                            "result": ingest_result.as_dict(),
                            "metadata": {
                                "generated_by": "operational_backbone.synthetic",
                            },
                        },
                        source="operational_backbone.synthetic",
                    )
                )

            sensory_snapshot: Mapping[str, Any] | None = None
            belief_state: BeliefState | None = None
            regime_signal: RegimeSignal | None = None
            belief_snapshot: BeliefSnapshot | None = None
            understanding_decision: UnderstandingDecision | None = None

            has_daily_frame = "daily_bars" in frames and not frames["daily_bars"].empty
            if has_daily_frame:
                daily_frame = frames["daily_bars"]
                belief_enabled = (self._belief_manager is not None) or (
                    self._understanding_router is not None
                )
                if belief_enabled:
                    if self._belief_manager is None:
                        if self._sensory_organ is None:
                            raise ValueError(
                                "sensory_organ must be provided when bootstrapping a belief manager",
                            )
                        if self._event_bus is None:
                            raise ValueError(
                                "event_bus must be provided when bootstrapping a belief manager",
                            )
                        belief_id = self._belief_id or f"operational.{symbol.lower()}"
                        manager, snapshot, belief_state, regime_signal = LiveBeliefManager.from_market_data(
                            market_data=daily_frame,
                            symbol=symbol,
                            belief_id=belief_id,
                            event_bus=self._event_bus,
                            organ=self._sensory_organ,
                        )
                        self._belief_manager = manager
                        sensory_snapshot = snapshot
                    else:
                        snapshot, belief_state, regime_signal = self._belief_manager.process_market_data(
                            daily_frame,
                            symbol=symbol,
                        )
                        sensory_snapshot = snapshot

                    if (
                        self._understanding_router is not None
                        and belief_state is not None
                        and regime_signal is not None
                    ):
                        feature_flags: Mapping[str, bool] | None = None
                        metadata = sensory_snapshot.get("metadata") if isinstance(sensory_snapshot, Mapping) else None
                        if isinstance(metadata, Mapping):
                            flags = metadata.get("feature_flags")
                            if isinstance(flags, Mapping):
                                feature_flags = {
                                    str(key): bool(value) for key, value in flags.items()
                                }
                        fast_weights_enabled = True
                        if isinstance(metadata, Mapping):
                            fast_flag = metadata.get("fast_weights_enabled")
                            if isinstance(fast_flag, bool):
                                fast_weights_enabled = fast_flag
                        belief_snapshot = BeliefSnapshot(
                            belief_id=belief_state.belief_id,
                            regime_state=regime_signal.regime_state,
                            features=dict(regime_signal.features),
                            metadata={
                                "symbol": belief_state.symbol,
                                "source": "operational_backbone",
                                "ingest_error": bool(ingest_error),
                            },
                            fast_weights_enabled=fast_weights_enabled,
                            feature_flags=feature_flags,
                        )
                        understanding_decision = self._understanding_router.route(belief_snapshot)
                elif self._sensory_organ is not None:
                    sensory_snapshot = await asyncio.to_thread(
                        self._sensory_organ.observe,
                        daily_frame,
                        symbol=symbol,
                    )

            consumer, owns_consumer = self._resolve_consumer()
            streaming_active = (
                self._streaming_task is not None
                and not self._streaming_task.done()
            )
            if poll_consumer and consumer is not None and not streaming_active:
                while True:
                    processed = await asyncio.to_thread(consumer.poll_once)
                    if not processed:
                        break
                    await asyncio.sleep(0)
                if self._event_bus is not None:
                    pending_queue = getattr(self._event_bus, "_queue", None)
                    if pending_queue is not None:
                        while not pending_queue.empty():
                            await asyncio.sleep(0)
                should_close = owns_consumer or (
                    self._auto_close_consumer
                    and self._streaming_task is None
                    and consumer is not self._streaming_consumer
                )
                if should_close:
                    consumer.close()

            ordered_dimensions: list[str] = [
                dimension
                for dimension, ingest_result in ingest_results.items()
                if dimension in expected_dimensions and ingest_result.rows_written > 0
            ]
            dimension_to_event: dict[str, Event] = {}
            for event in reversed(events):
                payload = event.payload if isinstance(event.payload, Mapping) else None
                if not isinstance(payload, Mapping):
                    continue
                result_blob = payload.get("result")
                if not isinstance(result_blob, Mapping):
                    continue
                dimension_token = result_blob.get("dimension")
                if isinstance(dimension_token, str) and dimension_token not in dimension_to_event:
                    dimension_to_event[dimension_token] = event

            final_events: list[Event] = []
            for dimension in ordered_dimensions:
                event = dimension_to_event.get(dimension)
                if event is None:
                    ingest_result = ingest_results[dimension]
                    event = Event(
                        type=primary_topic,
                        payload={
                            "result": ingest_result.as_dict(),
                            "metadata": {
                                "generated_by": "operational_backbone.synthetic",
                            },
                        },
                        source="operational_backbone.synthetic",
                    )
                final_events.append(event)

            events = final_events

            return OperationalBackboneResult(
                ingest_results=dict(ingest_results),
                frames=dict(frames),
                kafka_events=tuple(events),
                cache_metrics_before=dict(cache_before),
                cache_metrics_after_ingest=dict(cache_after_ingest),
                cache_metrics_after_fetch=dict(cache_after_fetch),
                sensory_snapshot=sensory_snapshot,
                belief_state=belief_state,
                regime_signal=regime_signal,
                belief_snapshot=belief_snapshot,
                understanding_decision=understanding_decision,
                ingest_error=ingest_error,
            )
        finally:
            if self._event_bus is not None:
                for handle in subscriptions:
                    self._event_bus.unsubscribe(handle)
                if bus_started_here:
                    await self._event_bus.stop()

    def _ensure_task_supervisor(self) -> TaskSupervisor:
        if self._task_supervisor is None:
            self._task_supervisor = TaskSupervisor(
                namespace="operational_backbone.kafka",
            )
            self._owns_task_supervisor = True
        return self._task_supervisor

    async def start_streaming(
        self,
        *,
        task_name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> asyncio.Task[None] | None:
        """Launch a supervised Kafka streaming loop feeding the event bus."""

        if self._streaming_task is not None:
            if not self._streaming_task.done():
                return self._streaming_task
            await self.stop_streaming()

        consumer = self._streaming_consumer
        owns_consumer = self._streaming_consumer_owned
        if consumer is None:
            if self._consumer is not None:
                consumer = self._consumer
                owns_consumer = False
            elif self._consumer_factory is not None:
                consumer = self._consumer_factory()
                owns_consumer = consumer is not None
            else:
                return None

        if consumer is None:
            return None

        if self._event_bus is not None and not self._event_bus.is_running():
            await self._event_bus.start()
            self._streaming_started_bus = True

        stop_event = asyncio.Event()
        supervisor = self._ensure_task_supervisor()

        base_metadata: dict[str, object] = dict(self._streaming_metadata or {})
        if metadata:
            base_metadata.update({str(k): v for k, v in metadata.items() if v is not None})

        run_task_name = task_name or self._streaming_task_name

        async def _runner() -> None:
            try:
                await consumer.run_forever(stop_event)
            finally:
                if owns_consumer and consumer is not self._consumer:
                    try:
                        consumer.close()
                    except Exception:  # pragma: no cover - defensive logging
                        logger.debug(
                            "Kafka ingest consumer close failed during streaming shutdown",
                            exc_info=True,
                        )

        task = supervisor.create(
            _runner(),
            name=run_task_name,
            metadata=base_metadata or None,
        )
        self._streaming_task = task
        self._streaming_stop_event = stop_event
        self._streaming_consumer = consumer
        self._streaming_consumer_owned = owns_consumer and consumer is not self._consumer
        self._subscribe_streaming_sensory()
        return task

    async def stop_streaming(self, *, cancel_timeout: float | None = None) -> None:
        """Stop the streaming task and clean up supervised resources."""

        task = self._streaming_task
        if task is None:
            return

        stop_event = self._streaming_stop_event
        if stop_event is not None:
            stop_event.set()

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._streaming_task = None
            self._streaming_stop_event = None

        if self._streaming_consumer is not None:
            if self._streaming_consumer_owned:
                try:
                    self._streaming_consumer.close()
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Kafka ingest consumer close failed after streaming",
                        exc_info=True,
                    )
            self._streaming_consumer = None
            self._streaming_consumer_owned = False

        if self._owns_task_supervisor and self._task_supervisor is not None:
            await self._task_supervisor.cancel_all()
            self._task_supervisor = None
            self._owns_task_supervisor = False

        self._unsubscribe_streaming_sensory()

        if self._event_bus is not None and self._streaming_started_bus:
            await self._event_bus.stop()
            self._streaming_started_bus = False

    async def shutdown(self) -> None:
        await self.stop_streaming()
        if self._manager_shutdown:
            return
        if not self._shutdown_manager_on_close:
            self._manager_shutdown = True
            return
        await self._manager.shutdown()
        self._manager_shutdown = True

    def _resolve_consumer(self) -> tuple[KafkaIngestEventConsumer | None, bool]:
        if self._streaming_consumer is not None:
            return self._streaming_consumer, False
        if self._consumer_factory is not None:
            consumer = self._consumer_factory()
            return consumer, True if consumer is not None else False
        if self._consumer is not None:
            return self._consumer, False
        return None, False

    @property
    def streaming_snapshots(self) -> Mapping[str, Mapping[str, Any]]:
        """Expose the latest sensory snapshots generated from streaming ingest events."""

        return dict(self._streaming_snapshots)

    async def _streaming_ingest_handler(self, event: Event) -> None:
        await self._handle_streaming_ingest_event(event)

    def _subscribe_streaming_sensory(self) -> None:
        if (
            not self._stream_sensory_from_kafka
            or self._sensory_organ is None
            or self._event_bus is None
        ):
            return
        if self._streaming_subscriptions:
            return

        handler = self._streaming_ingest_handler
        subscriptions: list[SubscriptionHandle] = []
        topics = self._event_topics or ("telemetry.ingest",)
        for topic in topics:
            try:
                subscriptions.append(self._event_bus.subscribe(topic, handler))
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "Failed to subscribe sensory stream handler to topic %s", topic
                )
        self._streaming_subscriptions = subscriptions if subscriptions else None

    def _unsubscribe_streaming_sensory(self) -> None:
        if not self._streaming_subscriptions or self._event_bus is None:
            self._streaming_subscriptions = None
            return

        for handle in self._streaming_subscriptions:
            try:
                self._event_bus.unsubscribe(handle)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "Failed to unsubscribe sensory stream handler for topic %s",
                    handle.event_type,
                )
        self._streaming_subscriptions = None

    async def _handle_streaming_ingest_event(self, event: Event) -> None:
        if (
            not self._stream_sensory_from_kafka
            or self._sensory_organ is None
            or self._manager_shutdown
        ):
            return

        payload = event.payload if isinstance(event.payload, Mapping) else None
        if not payload:
            return

        result_payload = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result_payload, Mapping):
            return

        dimension = str(result_payload.get("dimension") or "").strip().lower()
        if dimension != "daily_bars":
            return

        raw_symbols = result_payload.get("symbols")
        if isinstance(raw_symbols, Mapping):  # pragma: no cover - legacy defensive path
            symbols = [str(value).strip().upper() for value in raw_symbols.values() if value]
        else:
            symbols = [str(value).strip().upper() for value in (raw_symbols or ()) if value]
        if not symbols:
            return

        start = result_payload.get("start_ts")
        end = result_payload.get("end_ts")

        for symbol in symbols:
            snapshot = await self._produce_streaming_snapshot(
                symbol=symbol,
                start=start,
                end=end,
            )
            if snapshot is None:
                continue
            self._streaming_snapshots[symbol] = snapshot
            callback = self._sensory_snapshot_callback
            if callback is not None:
                try:
                    callback(snapshot)
                except Exception:  # pragma: no cover - defensive guard for callbacks
                    logger.exception(
                        "Sensory snapshot callback failed for symbol %s", symbol
                    )

    async def _produce_streaming_snapshot(
        self,
        *,
        symbol: str,
        start: object | None,
        end: object | None,
    ) -> Mapping[str, Any] | None:
        def _build_snapshot() -> Mapping[str, Any] | None:
            try:
                frame = self._manager.fetch_data(
                    symbol,
                    start=start,
                    end=end,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to fetch Timescale data for streaming sensory snapshot (%s): %s",
                    symbol,
                    exc,
                )
                return None

            if frame.empty:
                return None

            try:
                snapshot = self._sensory_organ.observe(frame, symbol=symbol)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Sensory organ failed to process streaming snapshot for %s",
                    symbol,
                )
                return None
            return snapshot

        return await asyncio.to_thread(_build_snapshot)


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
            task_supervisor=task_supervisor,
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
