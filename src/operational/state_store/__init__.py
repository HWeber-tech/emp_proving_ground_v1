"""Operational state-store adapters and helpers."""

from __future__ import annotations

from src.core.state_store import StateStore as StateStoreProtocol

from .adapters import InMemoryStateStore
from .registry import (
    get_global_state_store,
    get_state_store_sync,
    reset_global_state_store,
    set_global_state_store,
)

# Backwards-compatible export: historical code instantiates ``StateStore()``
# directly from this module. Re-export ``InMemoryStateStore`` under the legacy
# name so imports keep working while the decomposition continues.
StateStore = InMemoryStateStore


async def get_state_store() -> StateStoreProtocol:
    """Async alias preserved for historical call sites."""
    return await get_global_state_store()


__all__ = [
    "InMemoryStateStore",
    "StateStore",
    "get_global_state_store",
    "get_state_store",
    "get_state_store_sync",
    "reset_global_state_store",
    "set_global_state_store",
]
