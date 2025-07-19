"""
EMP State Store v1.1

Redis-based state management for the EMP system. Provides
persistent storage for population state, system state, and recovery data.
"""

import redis
import json
import logging
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from src.core.exceptions import StateStoreException

logger = logging.getLogger(__name__)


class StateStore:
    """Redis-based state store for EMP system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.redis_client: Optional[redis.Redis] = None
        self.key_prefixes = self.config.get('key_prefixes', {
            'population': 'emp:population:',
            'genome': 'emp:genome:',
            'strategy': 'emp:strategy:',
            'performance': 'emp:performance:',
            'risk': 'emp:risk:',
            'state': 'emp:state:',
            'events': 'emp:events:',
            'cache': 'emp:cache:'
        })
        self.ttl_settings = self.config.get('ttl', {
            'population': 3600,
            'genome': 86400,
            'strategy': 86400,
            'performance': 3600,
            'risk': 1800,
            'state': 300,
            'events': 7200,
            'cache': 600
        })
        self._connect()
        
    def _connect(self):
        """Connect to Redis."""
        try:
            connection_config = self.config.get('connection', {})
            self.redis_client = redis.Redis(
                host=connection_config.get('host', 'localhost'),
                port=connection_config.get('port', 6379),
                db=connection_config.get('db', 0),
                password=connection_config.get('password'),
                ssl=connection_config.get('ssl', False),
                decode_responses=False  # Keep as bytes for pickle
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis state store")
            
        except Exception as e:
            raise StateStoreException(f"Failed to connect to Redis: {e}")
            
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self.key_prefixes.get(prefix, 'emp:')}{identifier}"
        
    def _get_ttl(self, prefix: str) -> int:
        """Get TTL for a key prefix."""
        return self.ttl_settings.get(prefix, 3600)
        
    def store_population(self, population_id: str, population_data: Dict[str, Any]) -> bool:
        """Store population data."""
        try:
            key = self._get_key('population', population_id)
            data = pickle.dumps(population_data)
            ttl = self._get_ttl('population')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Stored population: {population_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing population {population_id}: {e}")
            return False
            
    def get_population(self, population_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve population data."""
        try:
            key = self._get_key('population', population_id)
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving population {population_id}: {e}")
            return None
            
    def store_genome(self, genome_id: str, genome_data: Dict[str, Any]) -> bool:
        """Store genome data."""
        try:
            key = self._get_key('genome', genome_id)
            data = pickle.dumps(genome_data)
            ttl = self._get_ttl('genome')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Stored genome: {genome_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing genome {genome_id}: {e}")
            return False
            
    def get_genome(self, genome_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve genome data."""
        try:
            key = self._get_key('genome', genome_id)
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving genome {genome_id}: {e}")
            return None
            
    def store_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """Store strategy data."""
        try:
            key = self._get_key('strategy', strategy_id)
            data = pickle.dumps(strategy_data)
            ttl = self._get_ttl('strategy')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Stored strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing strategy {strategy_id}: {e}")
            return False
            
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve strategy data."""
        try:
            key = self._get_key('strategy', strategy_id)
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving strategy {strategy_id}: {e}")
            return None
            
    def store_performance(self, performance_id: str, performance_data: Dict[str, Any]) -> bool:
        """Store performance data."""
        try:
            key = self._get_key('performance', performance_id)
            data = pickle.dumps(performance_data)
            ttl = self._get_ttl('performance')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Stored performance: {performance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance {performance_id}: {e}")
            return False
            
    def get_performance(self, performance_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve performance data."""
        try:
            key = self._get_key('performance', performance_id)
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving performance {performance_id}: {e}")
            return None
            
    def store_system_state(self, state_data: Dict[str, Any]) -> bool:
        """Store system state."""
        try:
            key = self._get_key('state', 'system')
            data = pickle.dumps(state_data)
            ttl = self._get_ttl('state')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug("Stored system state")
            return True
            
        except Exception as e:
            logger.error(f"Error storing system state: {e}")
            return False
            
    def get_system_state(self) -> Optional[Dict[str, Any]]:
        """Retrieve system state."""
        try:
            key = self._get_key('state', 'system')
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving system state: {e}")
            return None
            
    def store_event(self, event_id: str, event_data: Dict[str, Any]) -> bool:
        """Store event data."""
        try:
            key = self._get_key('events', event_id)
            data = pickle.dumps(event_data)
            ttl = self._get_ttl('events')
            
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Stored event: {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing event {event_id}: {e}")
            return False
            
    def get_events(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """Retrieve events matching pattern."""
        try:
            key_pattern = f"{self.key_prefixes['events']}{pattern}"
            keys = self.redis_client.keys(key_pattern)
            
            events = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    events.append(pickle.loads(data))
                    
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
            
    def cache_set(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Set cache data."""
        try:
            key = self._get_key('cache', cache_key)
            serialized_data = pickle.dumps(data)
            cache_ttl = ttl or self._get_ttl('cache')
            
            self.redis_client.setex(key, cache_ttl, serialized_data)
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache {cache_key}: {e}")
            return False
            
    def cache_get(self, cache_key: str) -> Optional[Any]:
        """Get cache data."""
        try:
            key = self._get_key('cache', cache_key)
            data = self.redis_client.get(key)
            
            if data:
                return pickle.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cache {cache_key}: {e}")
            return None
            
    def delete_key(self, prefix: str, identifier: str) -> bool:
        """Delete a key."""
        try:
            key = self._get_key(prefix, identifier)
            result = self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting key {prefix}:{identifier}: {e}")
            return False
            
    def clear_prefix(self, prefix: str) -> bool:
        """Clear all keys with a specific prefix."""
        try:
            key_pattern = f"{self.key_prefixes.get(prefix, 'emp:')}*"
            keys = self.redis_client.keys(key_pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys with prefix: {prefix}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error clearing prefix {prefix}: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get state store statistics."""
        try:
            stats = {}
            for prefix in self.key_prefixes:
                key_pattern = f"{self.key_prefixes[prefix]}*"
                keys = self.redis_client.keys(key_pattern)
                stats[f"{prefix}_count"] = len(keys)
                
            stats['total_keys'] = sum(stats.values())
            stats['redis_info'] = self.redis_client.info()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
            
    def health_check(self) -> bool:
        """Perform health check on Redis connection."""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
            
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed") 