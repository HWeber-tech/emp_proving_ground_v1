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
from ._event_bus_impl import (  # noqa: F401
    Event as Event,
    SubscriptionHandle as SubscriptionHandle,
    AsyncEventBus as AsyncEventBus,
    EventBus as EventBus,
    TopicBus as TopicBus,
    event_bus as event_bus,
    get_global_bus as get_global_bus,
    publish_event as publish_event,
    subscribe_to_event as subscribe_to_event,
    unsubscribe_from_event as unsubscribe_from_event,
    start_event_bus as start_event_bus,
    stop_event_bus as stop_event_bus,
)

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
