"""
Core State Store Port (Protocol)
===============================

Defines the minimal async key-value store interface used across the system.
Domain layers (thinking, sensory, etc.) should depend on this port only.

Concrete implementations live in higher layers (e.g., src/operational/state_store.py)
and must not be imported from here to preserve layering.
"""

from __future__ import annotations
from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """
    Async key-value store interface.

    Notes:
    - Implementations should be non-blocking where possible.
    - expiring keys are supported via the 'expire' parameter on set.
    """

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set a value with optional TTL in seconds."""
        ...

    async def get(self, key: str) -> Optional[str]:
        """Get a value for the given key, or None if missing/expired."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key, returning True if removed."""
        ...

    async def keys(self, pattern: str) -> List[str]:
        """Return keys matching a simple pattern (implementation-defined)."""
        ...

    async def clear(self) -> bool:
        """Clear all data."""
        ...


def is_state_store(obj: object) -> bool:
    """Runtime check helper for duck-typed implementations."""
    try:
        return isinstance(obj, StateStore)
    except Exception:
        return False