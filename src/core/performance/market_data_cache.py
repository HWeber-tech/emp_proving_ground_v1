import time
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


_global_cache: _InMemoryCache | None = None


def get_global_cache() -> _InMemoryCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = _InMemoryCache()
    return _global_cache