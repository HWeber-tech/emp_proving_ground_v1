"""
State Store for EMP Ultimate Architecture v1.1
Provides persistent state management for critical system data.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class StateStore:
    """Redis-based state store for critical system state."""
    
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.redis_client = None
        self._memory_store = {}
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host, 
                    port=port, 
                    db=db, 
                    password=password,
                    decode_responses=True
                )
                logger.info("StateStore configured for Redis")
            except Exception as e:
                logger.warning(f"Redis not available, using in-memory store: {e}")
        else:
            logger.warning("Redis not installed, using in-memory store")
            
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL."""
        try:
            if self.redis_client:
                serialized = json.dumps(value, default=str)
                if ttl:
                    result = await self.redis_client.setex(key, ttl, serialized)
                else:
                    result = await self.redis_client.set(key, serialized)
                return bool(result)
            else:
                # In-memory fallback
                self._memory_store[key] = {
                    'value': value,
                    'expires_at': datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
                }
                return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
            
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value by key."""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                # In-memory fallback
                if key in self._memory_store:
                    item = self._memory_store[key]
                    if item['expires_at'] is None or datetime.utcnow() < item['expires_at']:
                        return item['value']
                    else:
                        del self._memory_store[key]
                return None
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None
            
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            if self.redis_client:
                result = await self.redis_client.delete(key)
                return bool(result)
            else:
                # In-memory fallback
                if key in self._memory_store:
                    del self._memory_store[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            if self.redis_client:
                result = await self.redis_client.exists(key)
                return bool(result)
            else:
                return key in self._memory_store
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
            
    async def keys(self, pattern: str = "*") -> list:
        """Get all keys matching pattern."""
        try:
            if self.redis_client:
                result = await self.redis_client.keys(pattern)
                return list(result)
            else:
                return [k for k in self._memory_store.keys() if pattern == "*" or pattern in k]
        except Exception as e:
            logger.error(f"Error getting keys: {e}")
            return []
            
    async def flush_all(self) -> bool:
        """Clear all keys."""
        try:
            if self.redis_client:
                result = await self.redis_client.flushdb()
                return bool(result)
            else:
                self._memory_store.clear()
                return True
        except Exception as e:
            logger.error(f"Error flushing store: {e}")
            return False


# Global state store instance
state_store = StateStore()
