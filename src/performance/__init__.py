"""
Performance Module
==================

Performance optimization and caching utilities for the EMP trading system.
Public API is intentionally lightweight at import time; heavy numeric code is
lazy-imported to reduce import-time cost while preserving legacy public paths.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.performance.market_data_cache import (
    MarketDataCache as MarketDataCache,
)
from src.core.performance.market_data_cache import (
    get_global_cache as get_global_cache,
)

logger = logging.getLogger(__name__)
# Explicit public exports
__all__ = ["MarketDataCache", "VectorizedIndicators", "get_global_cache"]


_DEPRECATED_EXPORTS: dict[str, str] = {
    "GlobalCache": (
        "GlobalCache was removed; import MarketDataCache from "
        "src.core.performance.market_data_cache instead."
    )
}


def __getattr__(name: str) -> Any:
    if name == "VectorizedIndicators":
        from .vectorized_indicators import VectorizedIndicators

        return VectorizedIndicators
    if name in _DEPRECATED_EXPORTS:
        raise AttributeError(_DEPRECATED_EXPORTS[name])
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
