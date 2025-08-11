"""
Performance Module
==================

Performance optimization and caching utilities for the EMP trading system.
Provides high-performance caching and vectorized calculations.
"""

import logging
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class GlobalCache:
    """Simple in-memory cache for performance optimization."""
    
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0
    
    def get_market_data(self, category: str, subcategory: str, **kwargs) -> Optional[Any]:
        """Get cached market data."""
        key = f"{category}:{subcategory}:{str(sorted(kwargs.items()))}"
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def cache_market_data(self, category: str, subcategory: str, data: Any, **kwargs) -> None:
        """Cache market data."""
        key = f"{category}:{subcategory}:{str(sorted(kwargs.items()))}"
        self._cache[key] = data
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0,
            'size': len(self._cache)
        }


class VectorizedIndicators:
    """Vectorized technical indicators for performance."""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        if len(data) < period:
            return np.array([])
        return np.convolve(data, np.ones(period)/period, mode='valid')
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        if len(data) < period + 1:
            return np.array([])
        
        deltas = np.diff(data)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        
        rsi_values = [100 - 100 / (1 + rs)]
        
        for i in range(period + 1, len(data)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi_values.append(100 - 100 / (1 + rs))
        
        return np.array(rsi_values)
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> dict:
        """Bollinger Bands."""
        if len(data) < period:
            return {'upper': np.array([]), 'middle': np.array([]), 'lower': np.array([])}
        
        sma = VectorizedIndicators.sma(data, period)
        rolling_std = np.array([np.std(data[i:i+period]) for i in range(len(data) - period + 1)])
        
        upper = sma + (rolling_std * std_dev)
        lower = sma - (rolling_std * std_dev)
        
        return {'upper': upper, 'middle': sma, 'lower': lower}
    
    @staticmethod
    def calculate_all_indicators(market_data: dict, indicators: list = None) -> dict:
        """Calculate all requested indicators."""
        if indicators is None:
            indicators = ['sma', 'rsi', 'bb']
        
        results = {}
        close = np.array(market_data.get('close', []))
        
        if 'sma' in indicators:
            results['sma_20'] = VectorizedIndicators.sma(close, 20)
            results['sma_50'] = VectorizedIndicators.sma(close, 50)
        
        if 'rsi' in indicators:
            results['rsi_14'] = VectorizedIndicators.rsi(close, 14)
        
        if 'bb' in indicators:
            bb = VectorizedIndicators.bollinger_bands(close)
            results['bb_upper'] = bb['upper']
            results['bb_middle'] = bb['middle']
            results['bb_lower'] = bb['lower']
        
        return results


# Global instances
_global_cache = None


# Re-export canonical global cache accessor (deprecated local GlobalCache)
from core.performance.market_data_cache import get_global_cache as get_global_cache  # type: ignore


# Re-export for convenience
__all__ = [
    'GlobalCache',
    'VectorizedIndicators',
    'get_global_cache'
]
