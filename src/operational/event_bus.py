"""Operational EventBus shim.

This module re-exports the canonical EventBus and helpers from core.event_bus.
See docs/reports/CANONICALIZATION_PLAN.md for details.
"""

from core.event_bus import (
    EventBus as EventBus,
)
from core.event_bus import (
    event_bus as event_bus,
)
from core.event_bus import (
    publish_event as publish_event,
)
from core.event_bus import (
    start_event_bus as start_event_bus,
)
from core.event_bus import (
    stop_event_bus as stop_event_bus,
)
from core.event_bus import (
    subscribe_to_event as subscribe_to_event,
)
from core.event_bus import (
    unsubscribe_from_event as unsubscribe_from_event,
)

__all__ = [
    "EventBus",
    "event_bus",
    "publish_event",
    "subscribe_to_event",
    "unsubscribe_from_event",
    "start_event_bus",
    "stop_event_bus",
]
