"""
Performance Optimization Module
==============================

High-performance components for the EMP Proving Ground trading system.

This module provides:
- MarketDataCache: Ultra-fast Redis-based caching
- VectorizedIndicators: Optimized technical indicator calculations
- Performance monitoring and benchmarking utilities
"""

from .market_data_cache import MarketDataCache, get_global_cache
from .vectorized_indicators import VectorizedIndicators

__all__ = [
    'MarketDataCache',
    'get_global_cache',
    'VectorizedIndicators'
]
