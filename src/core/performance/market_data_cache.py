from __future__ import annotations

import math
import threading
import time
from typing import TypedDict, cast


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}
        self._expiry: dict[str, float] = {}

    def set(self, key: str, value: object, ttl_seconds: float | None = 300) -> None:
        if ttl_seconds is None:
            self._store[key] = value
            self._expiry.pop(key, None)
            return

        ttl = float(ttl_seconds)
        if math.isnan(ttl):
            raise ValueError("ttl_seconds cannot be NaN")

        if ttl <= 0:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return

        self._store[key] = value
        self._expiry[key] = time.time() + ttl

    def get(self, key: str, default: object | None = None) -> object | None:
        expires = self._expiry.get(key)
        if expires is not None and time.time() >= expires:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return default
        return self._store.get(key, default)


class MarketDataCache:
    """Simple in-memory market data cache with minimal API, plus legacy set/get support."""

    def __init__(self) -> None:
        # Symbol snapshot store
        self._data: dict[str, _Snapshot] = {}
        # Legacy generic KV store with TTL
        self._kv_store: dict[str, object] = {}
        self._kv_expiry: dict[str, float] = {}
        self._lock = threading.Lock()

    # Minimal snapshot API
    def put_snapshot(self, symbol: str, bid: float, ask: float, ts: float | str) -> None:
        """Store a snapshot for a symbol."""
        snapshot: _Snapshot = {"symbol": symbol, "bid": float(bid), "ask": float(ask), "ts": ts}
        with self._lock:
            self._data[symbol] = snapshot

    def get_snapshot(self, symbol: str) -> _Snapshot | None:
        """Retrieve the latest snapshot for a symbol."""
        with self._lock:
            snap = self._data.get(symbol)
            return cast(_Snapshot, dict(snap)) if snap is not None else None

    def maybe_get_mid(self, symbol: str) -> float | None:
        """Return (bid+ask)/2 for the symbol if available, else None."""
        snap = self.get_snapshot(symbol)
        if snap is None:
            return None
        try:
            bid = float(snap["bid"])
            ask = float(snap["ask"])
        except Exception:
            return None
        if not math.isfinite(bid) or not math.isfinite(ask):
            return None
        return (bid + ask) / 2.0

    # Legacy KV API (backward-compatible with _InMemoryCache)
    def set(self, key: str, value: object, ttl_seconds: float | None = 300) -> None:
        with self._lock:
            if ttl_seconds is None:
                self._kv_store[key] = value
                self._kv_expiry.pop(key, None)
                return

            ttl = float(ttl_seconds)
            if math.isnan(ttl):
                raise ValueError("ttl_seconds cannot be NaN")

            if ttl <= 0:
                self._kv_store.pop(key, None)
                self._kv_expiry.pop(key, None)
                return

            self._kv_store[key] = value
            self._kv_expiry[key] = time.time() + ttl

    def get(self, key: str, default: object | None = None) -> object | None:
        with self._lock:
            expires = self._kv_expiry.get(key)
            if expires is not None and time.time() >= expires:
                self._kv_store.pop(key, None)
                self._kv_expiry.pop(key, None)
                return default
            return self._kv_store.get(key, default)


_global_cache: MarketDataCache | None = None


def get_global_cache() -> MarketDataCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketDataCache()
    return _global_cache


# Backward compatibility: legacy in-memory KV cache alias
LegacyInMemoryCache = _InMemoryCache


class _Snapshot(TypedDict):
    symbol: str
    bid: float
    ask: float
    ts: float | str
