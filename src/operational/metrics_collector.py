"""
Metrics Collector System
========================

Collects and aggregates performance metrics across all EMP components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import time

from src.operational.state_store import StateStore
from src.core.events import EventBus

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system performance metrics."""
    
    def __init__(self, state_store: StateStore, event_bus: EventBus):
        self.state_store = state_store
        self.event_bus = event_bus
        self.metrics_buffer = []
        self.flush_interval = 60  # seconds
        self.is_running = False
        
    async def start(self) -> bool:
        """Start metrics collection."""
        if self.is_running:
            return True
            
        self.is_running = True
        asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")
        return True
    
    async def stop(self) -> bool:
        """Stop metrics collection."""
        self.is_running = False
        await self._flush_metrics()
        logger.info("Metrics collector stopped")
        return True
    
    async def record_metric(self, metric_type: str, value: float, 
                          tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        metric = {
            'metric_id': str(uuid.uuid4()),
            'type': metric_type,
            'value': value,
            'timestamp': datetime.utcnow().isoformat(),
            'tags': tags or {}
        }
        
        self.metrics_buffer.append(metric)
        
        # Flush if buffer is getting large
        if len(self.metrics_buffer) >= 100:
            await self._flush_metrics()
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def _flush_metrics(self) -> None:
        """Flush metrics to storage."""
        if not self.metrics_buffer:
            return
            
        try:
            # Store metrics in state store
            key = f"metrics:{datetime.utcnow().strftime('%Y%m%d')}"
            existing = await self.state_store.get(key)
            
            if existing:
                existing_metrics = eval(existing)
                existing_metrics.extend(self.metrics_buffer)
            else:
                existing_metrics = self.metrics_buffer
            
            await self.state_store.set(key, str(existing_metrics), expire=86400 * 7)
            
            # Emit metrics event
            await self.event_bus.emit('metrics_flushed', {
                'count': len(self.metrics_buffer),
                'types': list(set(m['type'] for m in self.metrics_buffer))
            })
            
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    async def get_metrics_summary(self, metric_type: str, 
                                hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for a type."""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            all_metrics = []
            
            # Collect metrics from recent days
            for i in range((hours // 24) + 1):
                date_str = (datetime.utcnow() - timedelta(days=i)).strftime('%Y%m%d')
                key = f"metrics:{date_str}"
                data = await self.state_store.get(key)
                
                if data:
                    metrics = eval(data)
                    filtered = [m for m in metrics 
                              if m['type'] == metric_type and 
                              datetime.fromisoformat(m['timestamp']) > cutoff]
                    all_metrics.extend(filtered)
            
            if not all_metrics:
                return {'count': 0, 'average': 0, 'min': 0, 'max': 0}
            
            values = [m['value'] for m in all_metrics]
            return {
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'last_value': values[-1] if values else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {'count': 0, 'average': 0, 'min': 0, 'max': 0}


# Global instance
_metrics_collector: Optional[MetricsCollector] = None


async def get_metrics_collector(state_store: StateStore, event_bus: EventBus) -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(state_store, event_bus)
    return _metrics_collector
