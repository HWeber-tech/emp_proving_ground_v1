"""
Event Bus for EMP Ultimate Architecture v1.1
Provides pub/sub communication between architectural layers.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from uuid import uuid4

from src.core.events import EventType, MarketUnderstanding, ContextPacket, TradeIntent, ExecutionReport, FitnessReport, GenomeUpdate, Telemetry

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for inter-layer communication."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Any] = []
        self._running = False
        
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from {event_type.value}")
            
    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribers."""
        event_type = self._get_event_type(event)
        self._event_history.append({
            'timestamp': datetime.utcnow(),
            'type': event_type,
            'event': event
        })
        
        logger.debug(f"Publishing {event_type.value}")
        
        if event_type in self._subscribers:
            tasks = []
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(callback(event))
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
                    
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
    def _get_event_type(self, event: Any) -> EventType:
        """Determine the event type from the event object."""
        event_class = event.__class__.__name__
        mapping = {
            'MarketUnderstanding': EventType.MARKET_UNDERSTANDING,
            'ContextPacket': EventType.CONTEXT_PACKET,
            'TradeIntent': EventType.TRADE_INTENT,
            'ExecutionReport': EventType.EXECUTION_REPORT,
            'FitnessReport': EventType.FITNESS_REPORT,
            'GenomeUpdate': EventType.GENOME_UPDATE,
            'Telemetry': EventType.TELEMETRY,
        }
        return mapping.get(event_class, EventType.TELEMETRY)
        
    def get_event_history(self, event_type: Optional[EventType] = None) -> List[Dict]:
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._event_history if e['type'] == event_type]
        return self._event_history
        
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# Global event bus instance
event_bus = EventBus()
