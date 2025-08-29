"""Deprecated namespace retained for compatibility."""
 
from __future__ import annotations

# Re-export canonical types from the core event bus implementation so
# callers can import from src.operational.bus for compatibility with
# existing import sites.
from src.core._event_bus_impl import AsyncEventBus as AsyncEventBus
from src.core._event_bus_impl import Event as Event  # noqa: F401

__all__ = ["Event", "AsyncEventBus"]
