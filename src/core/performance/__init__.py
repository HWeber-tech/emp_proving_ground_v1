"""
Performance Optimization Module
==============================

High-performance components for the EMP Proving Ground trading system.

This module provides:
- MarketDataCache: Ultra-fast Redis-based caching
- VectorizedIndicators: Optimized technical indicator calculations
- Performance monitoring and benchmarking utilities
"""

from __future__ import annotations

from .market_data_cache import MarketDataCache, get_global_cache

try:  # optional component
    from .vectorized_indicators import VectorizedIndicators  # type: ignore
except Exception:  # pragma: no cover

    class VectorizedIndicators:  # type: ignore
        pass


__all__ = ["MarketDataCache", "get_global_cache", "VectorizedIndicators"]
