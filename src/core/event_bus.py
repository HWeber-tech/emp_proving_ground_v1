"""
EMP Core Event Bus v1.1

Provides the event-driven communication infrastructure for inter-layer
communication in the EMP system. This is the backbone for the event-driven
architecture.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .events import BaseEvent, EventType, SensoryEvent, ThinkingEvent, TradingEvent, EvolutionEvent, GovernanceEvent, OperationalEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for EMP system communication."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[BaseEvent] = []
        self._max_history: int = 10000
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
    async def start(self):
        """Start the event bus."""
        self._running = True
        self._loop = asyncio.get_event_loop()
        logger.info("Event bus started")
        
    async def stop(self):
        """Stop the event bus."""
        self._running = False
        logger.info("Event bus stopped")
        
    def subscribe(self, event_type: EventType, callback: Callable[[BaseEvent], None]):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable[[BaseEvent], None]):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
                
    async def publish(self, event: BaseEvent):
        """Publish an event to all subscribers."""
        if not self._running:
            logger.warning("Event bus not running, event dropped")
            return
            
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
            
        # Notify subscribers
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    
        logger.debug(f"Published event: {event.event_type.value}")
        
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[BaseEvent]:
        """Get event history, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self._event_history if e.event_type == event_type]
        else:
            filtered = self._event_history
            
        return filtered[-limit:] if limit else filtered
        
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
event_bus = EventBus()


# Convenience functions for publishing events
async def publish_sensory_event(signal: Dict[str, Any], organ_id: str):
    """Publish a sensory event."""
    from .events import create_sensory_event
    event = create_sensory_event(signal, organ_id)
    await event_bus.publish(event)


async def publish_thinking_event(analysis_result: Dict[str, Any], pattern_id: str):
    """Publish a thinking event."""
    from .events import create_thinking_event
    event = create_thinking_event(analysis_result, pattern_id)
    await event_bus.publish(event)


async def publish_trading_event(decision: Dict[str, Any], strategy_id: str, symbol: str):
    """Publish a trading event."""
    from .events import create_trading_event
    event = create_trading_event(decision, strategy_id, symbol)
    await event_bus.publish(event)


async def publish_evolution_event(population_size: int, generation: int, best_fitness: float):
    """Publish an evolution event."""
    from .events import create_evolution_event
    event = create_evolution_event(population_size, generation, best_fitness)
    await event_bus.publish(event) 