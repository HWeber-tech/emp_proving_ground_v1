"""Helpers for managing process-wide state-store instances."""

from __future__ import annotations

from typing import Optional

from src.core.state_store import StateStore as StateStoreProtocol

from .adapters import InMemoryStateStore


_GLOBAL_STORE: StateStoreProtocol | None = None


def _ensure_store() -> StateStoreProtocol:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = InMemoryStateStore()
    return _GLOBAL_STORE


async def get_global_state_store() -> StateStoreProtocol:
    """Return the lazily-initialised global state store instance."""
    return _ensure_store()


def get_state_store_sync() -> StateStoreProtocol:
    """Synchronous accessor matching the async helper for legacy callers."""
    return _ensure_store()


def set_global_state_store(store: Optional[StateStoreProtocol]) -> None:
    """Override the process-global state store (useful for tests)."""
    global _GLOBAL_STORE
    _GLOBAL_STORE = store


def reset_global_state_store() -> None:
    """Reset the global state store so the next access creates a fresh instance."""
    set_global_state_store(None)


__all__ = [
    "get_global_state_store",
    "get_state_store_sync",
    "reset_global_state_store",
    "set_global_state_store",
]
