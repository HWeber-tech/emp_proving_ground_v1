"""Concrete state-store adapters used by operational modules."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Dict

from src.core.state_store import StateStore as StateStoreProtocol


class InMemoryStateStore(StateStoreProtocol):
    """Coroutine-friendly key/value store backed by an in-memory dictionary."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._expires: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"InMemoryStateStore(keys={len(self._store)})"

    async def set(self, key: str, value: str, expire: int | None = None) -> bool:
        async with self._lock:
            self._store[key] = value
            if expire is None:
                self._expires.pop(key, None)
            else:
                self._expires[key] = self._now() + timedelta(seconds=expire)
        return True

    async def get(self, key: str) -> str | None:
        async with self._lock:
            self._purge_expired()
            return self._store.get(key)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            self._expires.pop(key, None)
        return existed

    async def keys(self, pattern: str) -> list[str]:
        async with self._lock:
            self._purge_expired()
            needle = pattern or "*"
            return [key for key in self._store if fnmatch(key, needle)]

    async def clear(self) -> bool:
        async with self._lock:
            self._store.clear()
            self._expires.clear()
        return True

    async def purge_expired(self) -> int:
        async with self._lock:
            return self._purge_expired()

    def snapshot(self) -> dict[str, str]:
        return dict(self._store)

    def _now(self) -> datetime:
        return datetime.utcnow()

    def _purge_expired(self) -> int:
        removed = 0
        now = self._now()
        expired_keys = [key for key, expiry in self._expires.items() if expiry <= now]
        for key in expired_keys:
            self._store.pop(key, None)
            self._expires.pop(key, None)
            removed += 1
        return removed


__all__ = ["InMemoryStateStore"]
