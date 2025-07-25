"""Event bus for inter-component communication."""

import asyncio
import logging
from typing import Any, Callable, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents an event in the system."""
    type: str
    data: Any
    source: str = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


class EventBus:
    """Simple event bus for component communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events of a specific type."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
            except ValueError:
                pass
                
    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if event.type in self.subscribers:
            for callback in self.subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    
    def publish_sync(self, event_type: str, data: Any, source: str = None):
        """Publish a synchronous event."""
        event = Event(type=event_type, data=data, source=source)
        self.publish(event)
        
    async def publish_async(self, event_type: str, data: Any, source: str = None):
        """Publish an asynchronous event."""
        event = Event(type=event_type, data=data, source=source)
        self.publish(event)
