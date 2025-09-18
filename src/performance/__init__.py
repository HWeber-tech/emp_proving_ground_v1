"""
Performance Module
==================

Performance optimization and caching utilities for the EMP trading system.
Public API is intentionally lightweight at import time; heavy numeric code is
lazy-imported to reduce import-time cost while preserving legacy public paths.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.performance.market_data_cache import get_global_cache as get_global_cache

logger = logging.getLogger(__name__)


class GlobalCache:
    """Simple in-memory cache for performance optimization."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    def get_market_data(self, category: str, subcategory: str, **kwargs: Any) -> Optional[Any]:
        """Get cached market data."""
        key = f"{category}:{subcategory}:{str(sorted(kwargs.items()))}"
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def cache_market_data(self, category: str, subcategory: str, data: Any, **kwargs: Any) -> None:
        """Cache market data."""
        key = f"{category}:{subcategory}:{str(sorted(kwargs.items()))}"
        self._cache[key] = data

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
        }


# Explicit public exports
__all__ = ["GlobalCache", "VectorizedIndicators", "get_global_cache"]


def __getattr__(name: str) -> Any:
    # Lazy import to reduce import-time cost; preserves legacy public path.
    if name == "VectorizedIndicators":
        from .vectorized_indicators import VectorizedIndicators

        return VectorizedIndicators
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
