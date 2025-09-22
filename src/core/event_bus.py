"""
Canonical public API for EMP Event Bus.

This module is a typed, side-effect-free re-export shim that exposes the
Event Bus public interface from the private implementation module
src.core._event_bus_impl.

Do not import from src.core._event_bus_impl directly. Always import from:
  from src.core.event_bus import EventBus, AsyncEventBus, Event, TopicBus, ...

Runtime note:
- This module must not instantiate or start any loops; it only re-exports.
"""

from __future__ import annotations

# Re-export public types and functions from the private implementation module.
# Keeping exports explicit preserves stable API and helps type checkers.
from ._event_bus_impl import AsyncEventBus as AsyncEventBus
from ._event_bus_impl import Event as Event  # noqa: F401
from ._event_bus_impl import EventBus as EventBus
from ._event_bus_impl import EventBusStatistics as EventBusStatistics
from ._event_bus_impl import SubscriptionHandle as SubscriptionHandle
from ._event_bus_impl import TopicBus as TopicBus
from ._event_bus_impl import event_bus as event_bus
from ._event_bus_impl import get_global_bus as get_global_bus
from ._event_bus_impl import publish_event as publish_event
from ._event_bus_impl import set_event_bus_tracer as set_event_bus_tracer
from ._event_bus_impl import start_event_bus as start_event_bus
from ._event_bus_impl import stop_event_bus as stop_event_bus
from ._event_bus_impl import subscribe_to_event as subscribe_to_event
from ._event_bus_impl import unsubscribe_from_event as unsubscribe_from_event
from src.observability.tracing import (
    EventBusTracer,
    NullEventBusTracer,
    OpenTelemetryEventBusTracer,
    OpenTelemetrySettings,
    configure_event_bus_tracer,
    parse_opentelemetry_settings,
)

__all__ = [
    "Event",
    "SubscriptionHandle",
    "AsyncEventBus",
    "EventBus",
    "EventBusStatistics",
    "TopicBus",
    "event_bus",
    "get_global_bus",
    "publish_event",
    "subscribe_to_event",
    "unsubscribe_from_event",
    "start_event_bus",
    "stop_event_bus",
    "EventBusTracer",
    "NullEventBusTracer",
    "OpenTelemetryEventBusTracer",
    "OpenTelemetrySettings",
    "configure_event_bus_tracer",
    "parse_opentelemetry_settings",
    "set_event_bus_tracer",
]
