"""
EMP Event Bus M3 - Async-first with legacy-compatible Topic facade.

- Canonical model: AsyncEventBus with async publish(Event)
- Immutable Event dataclass
- Legacy Topic facade (TopicBus) for one release window:
  * publish_sync(topic, payload, source=None) - synchronous fan-out
  * subscribe_topic(topic, handler) - handler receives (type, payload)
  * Deprecated shims: publish(...), subscribe(...)
- Global accessor get_global_bus() returns a TopicBus bound to a process-global AsyncEventBus singleton.
  The global singleton is started on a dedicated background event loop thread to preserve legacy sync calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    cast,
)

from src.observability.tracing import EventBusTracer, NullEventBusTracer

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from src.runtime.task_supervisor import TaskSupervisor as _TaskSupervisor
else:  # pragma: no cover - executed at runtime, tested via behaviour
    try:
        from src.runtime.task_supervisor import TaskSupervisor as _TaskSupervisor  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - runtime without supervisor module
        _TaskSupervisor = None  # type: ignore[assignment]


def _new_task_supervisor(namespace: str) -> "_TaskSupervisor | None":
    """Instantiate a :class:`TaskSupervisor` if the runtime module is available."""

    if _TaskSupervisor is None:  # pragma: no cover - exercised when runtime unavailable
        return None
    return _TaskSupervisor(namespace=namespace)


# Immutable canonical event
@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any] | Any = None
    timestamp: float = field(default_factory=time.time)
    source: str | None = None


# Subscription handle
@dataclass(frozen=True)
class SubscriptionHandle:
    id: int
    event_type: str
    handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None]


@dataclass(frozen=True)
class EventBusStatistics:
    """Point-in-time diagnostics for the event bus."""

    running: bool
    loop_running: bool
    queue_size: int
    queue_capacity: int | None
    subscriber_count: int
    topic_subscribers: dict[str, int]
    published_events: int
    dropped_events: int
    handler_errors: int
    last_event_timestamp: float | None
    last_error_timestamp: float | None
    started_at: float | None
    uptime_seconds: float | None


_T = TypeVar("_T")


class TaskFactory(Protocol):
    """Callable responsible for spawning background tasks."""

    def __call__(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[_T]: ...


def _default_task_factory(
    coro: Coroutine[Any, Any, _T],
    *,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> asyncio.Task[_T]:
    """Spawn tasks using :func:`asyncio.create_task`.

    ``metadata`` is accepted for compatibility with :class:`TaskSupervisor`
    integrations but ignored by the default factory.
    """

    return asyncio.create_task(coro, name=name)


class AsyncEventBus:
    """
    Canonical async-first event bus.

    - subscribe(event_type, handler) -> SubscriptionHandle
      Set semantics: duplicate (event_type, handler) registration is ignored (returns same handle).
    - unsubscribe(handle) idempotent
    - async publish(Event) dispatches via internal worker/queue; fan-out concurrently; exceptions logged, not bubbled
    - publish_from_sync(Event) is thread-safe; schedules dispatch on the loop; warns once and returns None when not running
    - start()/stop() manage internal worker lifecycle
    """

    def __init__(
        self,
        *,
        task_factory: TaskFactory | None = None,
        tracer: EventBusTracer | None = None,
    ) -> None:
        self._subscribers: Dict[
            str, Set[Callable[[Event], None] | Callable[[Event], Awaitable[None]]]
        ] = {}
        self._pair_to_id: Dict[
            Tuple[str, Callable[[Event], None] | Callable[[Event], Awaitable[None]]], int
        ] = {}
        self._id_to_pair: Dict[
            int, Tuple[str, Callable[[Event], None] | Callable[[Event], Awaitable[None]]]
        ] = {}
        self._next_id: int = 1

        self._running: bool = False
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task_supervisor: "_TaskSupervisor | None" = None
        self._owns_task_supervisor: bool = False
        if task_factory is None:
            supervisor = _new_task_supervisor("event-bus")
            if supervisor is not None:
                self._task_supervisor = supervisor
                self._owns_task_supervisor = True
                self._task_factory = supervisor.create
            else:
                self._task_factory = _default_task_factory
        else:
            self._task_factory = task_factory
        self._tracer: EventBusTracer = tracer or NullEventBusTracer()

        # Warn-once flags
        self._warned_not_running_publish: bool = False
        self._warned_not_running_sync: bool = False
        self._warned_loop_issue_sync: bool = False
        self._warned_emit_deprecated: bool = False

        # Protects subscribe/unsubscribe maps from concurrent access across threads (handles only)
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._published_events_total: int = 0
        self._dropped_events_total: int = 0
        self._handler_errors_total: int = 0
        self._last_event_timestamp: float | None = None
        self._last_error_timestamp: float | None = None
        self._started_wall_time: float | None = None

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None],
    ) -> SubscriptionHandle:
        # Ensure set for event_type
        with self._lock:
            key = (event_type, handler)
            existing_id = self._pair_to_id.get(key)
            if existing_id is not None:
                return SubscriptionHandle(id=existing_id, event_type=event_type, handler=handler)

            # New registration
            sub_set = self._subscribers.setdefault(event_type, set())
            sub_set.add(handler)
            new_id = self._next_id
            self._next_id += 1
            self._pair_to_id[key] = new_id
            self._id_to_pair[new_id] = key
            logger.debug(
                "Subscribed handler %r to event_type %s as id=%d", handler, event_type, new_id
            )
            return SubscriptionHandle(id=new_id, event_type=event_type, handler=handler)

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        # Idempotent; accepts SubscriptionHandle
        with self._lock:
            pair = self._id_to_pair.pop(handle.id, None)
            if pair is None:
                return
            event_type, handler = pair
            self._pair_to_id.pop(pair, None)
            handlers = self._subscribers.get(event_type)
            if handlers is not None:
                handlers.discard(handler)
                if not handlers:
                    self._subscribers.pop(event_type, None)
            logger.debug(
                "Unsubscribed handler %r from event_type %s (id=%d)", handler, event_type, handle.id
            )

    async def publish(self, event: Event) -> None:
        """Enqueue event for async processing; logs and no-ops if not running."""
        if not self._running or self._loop is None:
            if not self._warned_not_running_publish:
                self._warned_not_running_publish = True
                logger.warning("AsyncEventBus.publish called while bus not running; event dropped")
            self._record_dropped_event()
            return
        metadata = {
            "mode": "async",
            "queue_size": self._queue.qsize(),
            "subscriber_count": self._subscriber_count(event.type),
        }
        with self._tracer.publish_span(
            event_type=event.type,
            event_source=event.source,
            metadata=metadata,
        ):
            self._record_published_event(event)
            await self._queue.put(event)

    def publish_from_sync(self, event: Event) -> int | None:
        """
        Thread-safe scheduling from non-async contexts.

        - Uses loop.call_soon_threadsafe to enqueue event into internal queue.
        - Returns a positive int (1) if schedule succeeded, None otherwise.
        - Warns once per process when the loop/bus is not running.
        """
        loop = self._loop
        if not self._running or loop is None or not loop.is_running():
            if not self._warned_not_running_sync:
                self._warned_not_running_sync = True
                logger.warning(
                    "AsyncEventBus.publish_from_sync called while loop/bus not running; no-op"
                )
            self._record_dropped_event()
            return None
        try:
            scheduled_count = self._subscriber_count(event.type)
            metadata = {
                "mode": "sync",
                "queue_size": self._queue.qsize(),
                "subscriber_count": scheduled_count,
            }
            with self._tracer.publish_span(
                event_type=event.type,
                event_source=event.source,
                metadata=metadata,
            ):
                loop.call_soon_threadsafe(self._queue.put_nowait, event)
                self._record_published_event(event)
            return scheduled_count
        except RuntimeError:
            if not self._warned_loop_issue_sync:
                self._warned_loop_issue_sync = True
                logger.warning("AsyncEventBus.publish_from_sync loop issue; returning None")
            self._record_dropped_event()
            return None

    async def start(self) -> None:
        """Start worker loop on current event loop."""
        if self._running:
            logger.warning("AsyncEventBus already running")
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._worker_task = self._spawn_task(
            self._worker(),
            name="AsyncEventBusWorker",
            metadata={"component": "event_bus", "task": "worker"},
        )
        logger.info("AsyncEventBus started")
        with self._metrics_lock:
            self._started_wall_time = time.time()

    async def stop(self) -> None:
        """Stop worker loop."""
        if not self._running:
            logger.warning("AsyncEventBus not running")
            return
        self._running = False
        task = self._worker_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        if self._owns_task_supervisor and self._task_supervisor is not None:
            try:
                await self._task_supervisor.cancel_all()
            except Exception:  # pragma: no cover - defensive clean-up logging
                logger.debug(
                    "Failed to cancel event bus supervised tasks", exc_info=True
                )
        logger.info("AsyncEventBus stopped")
        with self._metrics_lock:
            self._started_wall_time = None

    def is_running(self) -> bool:
        return self._running

    async def emit(
        self, topic: str, payload: dict[str, Any] | Any, source: str | None = None
    ) -> None:
        """Deprecated alias: await publish(Event(...))."""
        if not self._warned_emit_deprecated:
            self._warned_emit_deprecated = True
            warnings.warn(
                "AsyncEventBus.emit() is deprecated; use publish(Event(type=..., payload=...))",
                DeprecationWarning,
                stacklevel=2,
            )
        await self.publish(Event(type=topic, payload=payload, source=source))

    async def _worker(self) -> None:
        """Internal queue consumer."""
        assert self._loop is not None
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._fanout_event(event)
            except (
                Exception
            ):  # pragma: no cover - defensive; all exceptions should be handled internally
                logger.exception("Unexpected error during event fan-out")

    async def _fanout_event(self, event: Event) -> int:
        """Fan out to all matching handlers concurrently; log exceptions; return invoked count."""
        handlers_snapshot: list[Callable[[Event], None] | Callable[[Event], Awaitable[None]]]
        with self._lock:
            handlers_snapshot = list(self._subscribers.get(event.type, set()))
        if not handlers_snapshot:
            logger.debug("No subscribers for event type: %s", event.type)
            return 0

        tasks: list[asyncio.Task[Any]] = []
        dispatch_latency_ms = max((time.time() - event.timestamp) * 1000.0, 0.0)
        for handler in handlers_snapshot:
            handler_description = self._describe_handler(handler)
            coro = self._invoke_handler(handler, event, handler_description, dispatch_latency_ms)
            tasks.append(
                self._spawn_task(
                    coro,
                    name=f"event-handler:{event.type}",
                    metadata={
                        "component": "event_bus",
                        "task": "handler",
                        "event_type": event.type,
                        "handler": handler_description,
                        "dispatch_lag_ms": dispatch_latency_ms,
                    },
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for h, res in zip(handlers_snapshot, results):
            if isinstance(res, Exception):
                # Log explicitly at ERROR level (not relying on active exc state for logger.exception)
                logger.error("Error in handler %r for event type %s", h, event.type)
                self._record_handler_error()
        return len(handlers_snapshot)

    async def _invoke_handler(
        self,
        handler: Callable[[Event], None] | Callable[[Event], Awaitable[None]],
        event: Event,
        handler_description: str,
        dispatch_lag_ms: float,
    ) -> None:
        metadata = {"dispatch_lag_ms": dispatch_lag_ms}
        with self._tracer.handler_span(
            event_type=event.type,
            handler_name=handler_description,
            metadata=metadata,
        ):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Log and do not bubble exceptions from handlers
                logger.error(
                    "Error in handler %s for event type %s",
                    handler_description,
                    event.type,
                )
                self._record_handler_error()

    def _record_published_event(self, event: Event) -> None:
        with self._metrics_lock:
            self._published_events_total += 1
            self._last_event_timestamp = event.timestamp

    def _record_dropped_event(self) -> None:
        with self._metrics_lock:
            self._dropped_events_total += 1

    def _record_handler_error(self) -> None:
        with self._metrics_lock:
            self._handler_errors_total += 1
            self._last_error_timestamp = time.time()

    def set_task_factory(self, task_factory: TaskFactory | None) -> None:
        """Override the task factory used for worker and handler tasks."""

        if task_factory is None:
            if self._task_supervisor is not None and self._owns_task_supervisor:
                self._task_factory = self._task_supervisor.create
                return
            supervisor = _new_task_supervisor("event-bus")
            if supervisor is not None:
                self._task_supervisor = supervisor
                self._owns_task_supervisor = True
                self._task_factory = supervisor.create
            else:
                self._task_supervisor = None
                self._owns_task_supervisor = False
                self._task_factory = _default_task_factory
            return

        self._task_supervisor = None
        self._owns_task_supervisor = False
        self._task_factory = task_factory

    def set_tracer(self, tracer: EventBusTracer | None) -> None:
        """Override the tracer used for publish and handler spans."""

        self._tracer = tracer or NullEventBusTracer()

    def _spawn_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        try:
            return self._task_factory(coro, name=name, metadata=metadata)
        except TypeError:
            # Support legacy factories that only accept (coro, name)
            legacy_factory = cast(
                Callable[[Coroutine[Any, Any, Any]], asyncio.Task[Any]],
                self._task_factory,
            )
            task = legacy_factory(coro)
            if name is not None:
                try:
                    task.set_name(name)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.debug("Failed to assign task name %s", name, exc_info=exc)
            return task

    @staticmethod
    def _describe_handler(
        handler: Callable[[Event], None] | Callable[[Event], Awaitable[None]],
    ) -> str:
        try:
            description = repr(handler)
        except Exception:  # pragma: no cover - defensive
            return "<handler>"
        if len(description) > 120:
            return f"{description[:117]}..."
        return description

    def _subscriber_count(self, event_type: str) -> int:
        with self._lock:
            handlers = self._subscribers.get(event_type)
            return len(handlers) if handlers else 0

    def get_statistics(self) -> EventBusStatistics:
        """Return a diagnostic snapshot of the current event bus state."""

        with self._lock:
            topic_counts: dict[str, int] = {
                topic: len(handlers) for topic, handlers in self._subscribers.items()
            }

        with self._metrics_lock:
            published_total = self._published_events_total
            dropped_total = self._dropped_events_total
            handler_errors = self._handler_errors_total
            last_event_ts = self._last_event_timestamp
            last_error_ts = self._last_error_timestamp
            started_wall_time = self._started_wall_time

        loop_running = self._loop is not None and self._loop.is_running()
        queue_capacity = self._queue.maxsize if self._queue.maxsize > 0 else None
        uptime_seconds: float | None = None
        if started_wall_time is not None and self._running and loop_running:
            uptime_seconds = max(time.time() - started_wall_time, 0.0)

        return EventBusStatistics(
            running=self._running,
            loop_running=loop_running,
            queue_size=self._queue.qsize(),
            queue_capacity=queue_capacity,
            subscriber_count=sum(topic_counts.values()),
            topic_subscribers=topic_counts,
            published_events=published_total,
            dropped_events=dropped_total,
            handler_errors=handler_errors,
            last_event_timestamp=last_event_ts,
            last_error_timestamp=last_error_ts,
            started_at=started_wall_time,
            uptime_seconds=uptime_seconds,
        )


# Export alias for continuity
EventBus = AsyncEventBus


# Legacy-compatible topic facade
class TopicBus:
    """
    Compatibility facade over AsyncEventBus.
    - publish_sync(topic, payload, source=None) synchronously fans out to subscribers by executing
      the handlers on the AsyncEventBus loop using run_coroutine_threadsafe, returning the invoked count.
    - subscribe_topic(topic, handler) adapts the handler to receive (event.type, event.payload).
      Supports both sync and async handler callables; awaits if coroutine returned.
    - Deprecated wrappers:
      * publish(topic, payload) - warns once; calls publish_sync
      * subscribe(topic, handler) - warns once; adapts payload-only signature
    - unsubscribe(handle) delegates to underlying bus; idempotent
    """

    _warned_publish_once: bool = False
    _warned_subscribe_once: bool = False

    def __init__(self, bus: AsyncEventBus) -> None:
        self._bus = bus
        # Preserve adapter identity to avoid duplicate registrations for same (topic, handler)
        self._adapter_map: Dict[
            Tuple[str, Callable[..., object]],
            Callable[[Event], None] | Callable[[Event], Awaitable[None]],
        ] = {}

    def publish_sync(
        self, topic: str, payload: dict[str, Any] | Any, source: str | None = None
    ) -> int | None:
        loop = self._bus._loop
        if not self._bus.is_running() or loop is None or not loop.is_running():
            logger.warning("TopicBus.publish_sync called while bus not running; no-op")
            self._bus._record_dropped_event()
            return None

        event = Event(type=topic, payload=payload, source=source, timestamp=time.time())
        subscriber_count = self._bus._subscriber_count(topic)
        metadata = {
            "mode": "topic_sync",
            "queue_size": self._bus._queue.qsize(),
            "subscriber_count": subscriber_count,
        }
        with self._bus._tracer.publish_span(
            event_type=topic,
            event_source=source,
            metadata=metadata,
        ):
            self._bus._record_published_event(event)
            # Synchronously fan out on the event loop thread to preserve legacy semantics.
            try:
                fut = asyncio.run_coroutine_threadsafe(self._bus._fanout_event(event), loop)
                count = fut.result(timeout=2.0)
                # Ensure an int is returned
                return int(count)
            except FuturesTimeoutError:
                logger.warning("TopicBus.publish_sync timed out waiting for handler fan-out")
                return 0
            except Exception:
                logger.exception("TopicBus.publish_sync unexpected error during fan-out")
                return 0

    def subscribe_topic(
        self,
        topic: str,
        handler: Callable[[str, Any], None | Awaitable[None]],
    ) -> SubscriptionHandle:
        key = (topic, handler)
        adapter_fn = self._adapter_map.get(key)
        if adapter_fn is None:

            def _adapter(ev: Event) -> None | Awaitable[None]:
                res = handler(ev.type, ev.payload)
                if asyncio.iscoroutine(res):
                    return res
                return None

            adapter_typed = cast(
                Callable[[Event], None] | Callable[[Event], Awaitable[None]], _adapter
            )
            adapter_fn = adapter_typed
            self._adapter_map[key] = adapter_typed
        return self._bus.subscribe(topic, adapter_fn)

    # Deprecated wrappers

    def publish(self, topic: str, payload: dict[str, Any] | Any) -> int | None:
        if not TopicBus._warned_publish_once:
            TopicBus._warned_publish_once = True
            warnings.warn(
                "TopicBus.publish() is deprecated; use publish_sync(topic, payload, source=None)",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.publish_sync(topic, payload)

    def subscribe(
        self, topic: str, handler: Callable[[Any], None | Awaitable[None]]
    ) -> SubscriptionHandle:
        if not TopicBus._warned_subscribe_once:
            TopicBus._warned_subscribe_once = True
            warnings.warn(
                "TopicBus.subscribe() is deprecated; use subscribe_topic(topic, handler)",
                DeprecationWarning,
                stacklevel=2,
            )

        # Adapt payload-only legacy signature to (type, payload)
        def adapter(_type: str, payload: object) -> None | Awaitable[None]:
            res = handler(payload)
            if asyncio.iscoroutine(res):
                return res
            return None

        return self.subscribe_topic(topic, adapter)

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        self._bus.unsubscribe(handle)


# -------------------------
# Global singleton and helpers
# -------------------------

# Process-global AsyncEventBus singleton
event_bus: AsyncEventBus = AsyncEventBus()

# Global TopicBus facade (lazy)
_GLOBAL_TOPIC_BUS: Optional[TopicBus] = None

# Background loop thread management for global bus to preserve legacy sync publish() semantics
_BG_THREAD: Optional[threading.Thread] = None
_BG_INIT_LOCK = threading.Lock()


def _ensure_global_bus_running_in_background() -> None:
    global _BG_THREAD
    if event_bus.is_running() and event_bus._loop is not None and event_bus._loop.is_running():
        return

    with _BG_INIT_LOCK:
        if _BG_THREAD is not None and _BG_THREAD.is_alive():
            # Thread already running; fall through to readiness wait below
            pass
        else:

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(event_bus.start())
                    loop.run_forever()
                finally:
                    try:
                        if event_bus.is_running():
                            loop.run_until_complete(event_bus.stop())
                    except Exception as exc:  # pragma: no cover - defensive shutdown logging
                        logger.warning("Failed to shutdown event bus cleanly", exc_info=exc)
                    loop.close()

            _BG_THREAD = threading.Thread(
                target=_runner, name="AsyncEventBusLoopThread", daemon=True
            )
            _BG_THREAD.start()

    # Briefly wait for the loop to be ready to avoid race on immediate publish_sync()
    for _ in range(100):
        loop = event_bus._loop
        if event_bus.is_running() and loop is not None and loop.is_running():
            break
        time.sleep(0.01)


def get_global_bus() -> TopicBus:
    """Return TopicBus facade bound to the process-global AsyncEventBus singleton."""
    global _GLOBAL_TOPIC_BUS
    if _GLOBAL_TOPIC_BUS is None:
        _GLOBAL_TOPIC_BUS = TopicBus(event_bus)
    _ensure_global_bus_running_in_background()
    return _GLOBAL_TOPIC_BUS


# Convenience functions reusing the global AsyncEventBus singleton (re-exported by operational shims)
def set_event_bus_tracer(tracer: EventBusTracer | None) -> None:
    event_bus.set_tracer(tracer)


async def publish_event(event: Event) -> None:
    await event_bus.publish(event)


def subscribe_to_event(
    event_type: str,
    callback: Callable[[Event], Awaitable[None]] | Callable[[Event], None],
) -> SubscriptionHandle:
    return event_bus.subscribe(event_type, callback)


def unsubscribe_from_event(event_type: str, callback: Callable[[Event], object]) -> None:
    # Best-effort legacy removal by (event_type, callback)
    with event_bus._lock:
        sub_id = event_bus._pair_to_id.get(
            (
                event_type,
                cast(Callable[[Event], None] | Callable[[Event], Awaitable[None]], callback),
            )
        )
    if sub_id is not None:
        handle = SubscriptionHandle(
            id=sub_id,
            event_type=event_type,
            handler=cast(Callable[[Event], None] | Callable[[Event], Awaitable[None]], callback),
        )
        event_bus.unsubscribe(handle)


async def start_event_bus() -> None:
    await event_bus.start()


async def stop_event_bus() -> None:
    await event_bus.stop()


__all__ = [
    "Event",
    "SubscriptionHandle",
    "AsyncEventBus",
    "EventBus",
    "TopicBus",
    "event_bus",
    "get_global_bus",
    "publish_event",
    "subscribe_to_event",
    "unsubscribe_from_event",
    "start_event_bus",
    "stop_event_bus",
]
