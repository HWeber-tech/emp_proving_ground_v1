"""
State Store - Simple In-Memory Implementation
===========================================

Provides state management for testing Phase 3 systems.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


class StateStore:
    """Simple in-memory state store for testing."""

    def __init__(self):
        self._store = {}
        self._expires = {}

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set a value in the store."""
        self._store[key] = value
        if expire:
            self._expires[key] = datetime.utcnow() + timedelta(seconds=expire)
        return True

    async def get(self, key: str) -> Optional[str]:
        """Get a value from the store."""
        # Check expiration
        if key in self._expires:
            if datetime.utcnow() > self._expires[key]:
                self._store.pop(key, None)
                self._expires.pop(key, None)
                return None

        return self._store.get(key)

    async def delete(self, key: str) -> bool:
        """Delete a key from the store."""
        self._store.pop(key, None)
        self._expires.pop(key, None)
        return True

    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching a pattern."""
        # Simple pattern matching - just contains check
        matching = []
        for key in self._store.keys():
            if pattern.replace("*", "") in key:
                matching.append(key)
        return matching

    async def clear(self) -> bool:
        """Clear all data."""
        self._store.clear()
        self._expires.clear()
        return True


# Global instance
_state_store: Optional[StateStore] = None


async def get_state_store() -> StateStore:
    """Get or create global state store instance."""
    global _state_store
    if _state_store is None:
        _state_store = StateStore()
    return _state_store
