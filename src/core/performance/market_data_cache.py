import time
import threading
from typing import Any, Dict


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        self._store[key] = value
        self._expiry[key] = time.time() + ttl_seconds

    def get(self, key: str, default: Any = None) -> Any:
        expires = self._expiry.get(key)
        if expires is not None and time.time() > expires:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return default
        return self._store.get(key, default)

class MarketDataCache:
    """Simple in-memory market data cache with minimal API, plus legacy set/get support."""
    def __init__(self) -> None:
        # Symbol snapshot store
        self._data: Dict[str, Dict[str, Any]] = {}
        # Legacy generic KV store with TTL
        self._kv_store: Dict[str, Any] = {}
        self._kv_expiry: Dict[str, float] = {}
        self._lock = threading.Lock()

    # Minimal snapshot API
    def put_snapshot(self, symbol: str, bid: float, ask: float, ts: float | str) -> None:
        """Store a snapshot for a symbol."""
        snapshot = {"symbol": symbol, "bid": float(bid), "ask": float(ask), "ts": ts}
        with self._lock:
            self._data[symbol] = snapshot

    def get_snapshot(self, symbol: str) -> Dict[str, Any] | None:
        """Retrieve the latest snapshot for a symbol."""
        with self._lock:
            snap = self._data.get(symbol)
            return dict(snap) if snap is not None else None

    def maybe_get_mid(self, symbol: str) -> float | None:
        """Return (bid+ask)/2 for the symbol if available, else None."""
        snap = self.get_snapshot(symbol)
        if snap is None:
            return None
        try:
            return (float(snap["bid"]) + float(snap["ask"])) / 2.0
        except Exception:
            return None

    # Legacy KV API (backward-compatible with _InMemoryCache)
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        with self._lock:
            self._kv_store[key] = value
            self._kv_expiry[key] = time.time() + ttl_seconds

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            expires = self._kv_expiry.get(key)
            if expires is not None and time.time() > expires:
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
LegacyInMemoryCache = _InMemoryCache  # type: ignore