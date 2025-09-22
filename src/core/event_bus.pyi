from __future__ import annotations

import asyncio

from typing import Awaitable, Callable, ContextManager, Mapping, Protocol

class Event:
    type: str
    payload: dict[str, object] | object
    timestamp: float
    source: str | None

class SubscriptionHandle:
    id: int
    event_type: str
    handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None]

class EventBusStatistics:
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

class TaskFactory(Protocol):
    def __call__(
        self,
        coro: Awaitable[object],
        *,
        name: str | None = ...,
        metadata: Mapping[str, object] | None = ...,
    ) -> asyncio.Task[object]: ...

class EventBusTracer(Protocol):
    def publish_span(
        self,
        *,
        event_type: str,
        event_source: str | None,
        metadata: Mapping[str, object] | None = ...,
    ) -> ContextManager[object]: ...
    def handler_span(
        self,
        *,
        event_type: str,
        handler_name: str,
        metadata: Mapping[str, object] | None = ...,
    ) -> ContextManager[object]: ...

class NullEventBusTracer(EventBusTracer): ...
class OpenTelemetryEventBusTracer(EventBusTracer): ...

class OpenTelemetrySettings:
    enabled: bool
    service_name: str
    environment: str | None
    endpoint: str | None
    headers: Mapping[str, str] | None
    timeout: float | None
    console_exporter: bool

class AsyncEventBus:
    def __init__(
        self,
        *,
        task_factory: TaskFactory | None = ...,
        tracer: EventBusTracer | None = ...,
    ) -> None: ...
    def subscribe(
        self, event_type: str, handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None]
    ) -> SubscriptionHandle: ...
    def unsubscribe(self, handle: SubscriptionHandle) -> None: ...
    async def publish(self, event: Event) -> None: ...
    def publish_from_sync(self, event: Event) -> int | None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    async def emit(
        self, topic: str, payload: dict[str, object] | object, source: str | None = None
    ) -> None: ...
    def get_statistics(self) -> EventBusStatistics: ...
    def set_task_factory(self, task_factory: TaskFactory | None) -> None: ...
    def set_tracer(self, tracer: EventBusTracer | None) -> None: ...

EventBus = AsyncEventBus

class TopicBus:
    def publish_sync(
        self, topic: str, payload: dict[str, object] | object, source: str | None = None
    ) -> int | None: ...
    def subscribe_topic(
        self, topic: str, handler: Callable[[str, object], None | Awaitable[None]]
    ) -> SubscriptionHandle: ...
    def publish(self, topic: str, payload: dict[str, object] | object) -> int | None: ...
    def subscribe(
        self, topic: str, handler: Callable[[object], None | Awaitable[None]]
    ) -> SubscriptionHandle: ...
    def unsubscribe(self, handle: SubscriptionHandle) -> None: ...

event_bus: AsyncEventBus

def get_global_bus() -> TopicBus: ...
async def publish_event(event: Event) -> None: ...
def subscribe_to_event(
    event_type: str, callback: Callable[[Event], Awaitable[None]] | Callable[[Event], None]
) -> SubscriptionHandle: ...
def unsubscribe_from_event(
    event_type: str, callback: Callable[[Event], Awaitable[None]] | Callable[[Event], None]
) -> None: ...
async def start_event_bus() -> None: ...
async def stop_event_bus() -> None: ...
def parse_opentelemetry_settings(extras: Mapping[str, str] | None) -> OpenTelemetrySettings: ...
def configure_event_bus_tracer(settings: OpenTelemetrySettings) -> EventBusTracer | None: ...
def set_event_bus_tracer(tracer: EventBusTracer | None) -> None: ...
