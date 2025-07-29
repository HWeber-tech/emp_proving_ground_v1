"""
High-Performance Market Data Cache
=================================

Redis-based caching layer for ultra-fast market data access and reduced memory usage.
Optimized for sub-millisecond access times and production-grade reliability.
"""

import redis
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)


class MarketDataCache:
    """Ultra-fast Redis-based market data caching system with fallback support."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0,
                 password: Optional[str] = None, ttl: int = 300, max_connections: int = 50):
        """
        Initialize market data cache with connection pooling.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            ttl: Default TTL in seconds
            max_connections: Maximum Redis connections in pool
        """
        self.ttl = ttl
        self.host = host
        self.port = port
        
        # Initialize Redis connection pool
        try:
            self.redis_pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                max_connections=max_connections,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            self.redis_client.ping()
            logger.info("MarketDataCache connected to Redis with connection pooling")
            self.use_redis = True
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.redis_client = None
            self.use_redis = False
            self.fallback_cache = {}
    
    def _generate_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate consistent cache key for market data."""
        key_parts = [symbol, data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return ":".join(key_parts)
    
    def _serialize_array(self, array: np.ndarray) -> bytes:
        """Serialize numpy array using efficient binary format."""
        return pickle.dumps({
            'data': array.tobytes(),
            'shape': array.shape,
            'dtype': str(array.dtype)
        })
    
    def _deserialize_array(self, data: bytes) -> np.ndarray:
        """Deserialize numpy array from binary format."""
        try:
            cached = pickle.loads(data)
            return np.frombuffer(cached['data'], dtype=np.dtype(cached['dtype'])).reshape(cached['shape'])
        except Exception as e:
            logger.error(f"Failed to deserialize array: {e}")
            return None
    
    def cache_market_data(self, symbol: str, data_type: str, data: np.ndarray,
                         ttl: Optional[int] = None, **kwargs) -> bool:
        """Cache market data with automatic serialization."""
        if not self.use_redis:
            key = self._generate_key(symbol, data_type, **kwargs)
            self.fallback_cache[key] = {
                'data': data.copy(),
                'timestamp': datetime.now(),
                'ttl': ttl or self.ttl
            }
            return True
        
        try:
            key = self._generate_key(symbol, data_type, **kwargs)
            serialized_data = self._serialize_array(data)
            self.redis_client.setex(key, ttl or self.ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Failed to cache market data: {e}")
            return False
    
    def get_market_data(self, symbol: str, data_type: str, **kwargs) -> Optional[np.ndarray]:
        """Retrieve cached market data with automatic deserialization."""
        if not self.use_redis:
            key = self._generate_key(symbol, data_type, **kwargs)
            if key in self.fallback_cache:
                entry = self.fallback_cache[key]
                if datetime.now() - entry['timestamp'] < timedelta(seconds=entry['ttl']):
                    return entry['data']
                else:
                    del self.fallback_cache[key]
            return None
        
        try:
            key = self._generate_key(symbol, data_type, **kwargs)
            cached_data = self.redis_client.get(key)
            if cached_data is None:
                return None
            return self._deserialize_array(cached_data)
        except Exception as e:
            logger.error(f"Failed to retrieve market data: {e}")
            return None
    
    def cache_indicator(self, symbol: str, indicator_name: str,
                       indicator_data: np.ndarray, params: Dict[str, Any]) -> bool:
        """Cache technical indicator results to avoid recalculation."""
        return self.cache_market_data(
            symbol,
            f"indicator:{indicator_name}",
            indicator_data,
            **params
        )
    
    def get_indicator(self, symbol: str, indicator_name: str,
                     params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Retrieve cached indicator results."""
        return self.get_market_data(
            symbol,
            f"indicator:{indicator_name}",
            **params
        )
    
    def cache_price_levels(self, symbol: str, levels: Dict[str, float],
                          level_type: str = "support_resistance") -> bool:
        """Cache calculated price levels."""
        key = f"levels:{symbol}:{level_type}"
        
        if not self.use_redis:
            self.fallback_cache[key] = {
                'data': levels,
                'timestamp': datetime.now(),
                'ttl': self.ttl
            }
            return True
        
        try:
            self.redis_client.setex(key, self.ttl, json.dumps(levels))
            return True
        except Exception as e:
            logger.error(f"Failed to cache price levels: {e}")
            return False
    
    def get_price_levels(self, symbol: str,
                        level_type: str = "support_resistance") -> Optional[Dict[str, float]]:
        """Retrieve cached price levels."""
        key = f"levels:{symbol}:{level_type}"
        
        if not self.use_redis:
            if key in self.fallback_cache:
                entry = self.fallback_cache[key]
                if datetime.now() - entry['timestamp'] < timedelta(seconds=entry['ttl']):
                    return entry['data']
            return None
        
        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Failed to retrieve price levels: {e}")
            return None
    
    def invalidate_symbol(self, symbol: str) -> bool:
        """Invalidate all cached data for a symbol."""
        if not self.use_redis:
            keys_to_delete = [k for k in self.fallback_cache.keys() if k.startswith(symbol)]
            for key in keys_to_delete:
                del self.fallback_cache[key]
            return True
        
        try:
            pattern = f"{symbol}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate symbol cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.use_redis:
            return {
                'type': 'in-memory',
                'size': len(self.fallback_cache),
                'host': self.host,
                'port': self.port,
                'status': 'active'
            }
        
        try:
            info = self.redis_client.info()
            total_ops = info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)
            hit_rate = info.get('keyspace_hits', 0) / max(total_ops, 1)
            
            return {
                'type': 'redis',
                'connected': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'keys': info.get('db0', {}).get('keys', 0),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': round(hit_rate, 4),
                'host': self.host,
                'port': self.port,
                'status': 'active'
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                'type': 'redis',
                'connected': False,
                'error': str(e),
                'status': 'error'
            }


# Global cache instance for easy access
_global_cache = None


def get_global_cache() -> MarketDataCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MarketDataCache()
    return _global_cache


if __name__ == "__main__":
    # Production cache initialization
    cache = MarketDataCache()
    print(f"MarketDataCache initialized with {cache.max_size} max entries")
    print(f"Cache stats: {cache.get_cache_stats()}")
    print("Use real market data for testing cache functionality")
