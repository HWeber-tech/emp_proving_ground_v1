"""
EMP Event Bus v1.1

Asyncio-based event bus for inter-layer communication
in the EMP Ultimate Architecture v1.1.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

try:
    from .events import BaseEvent, EventType  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal smoke tests
    class EventType(str):  # type: ignore
        value: str

        def __new__(cls, value: str = ""):
            obj = str.__new__(cls, value)
            obj.value = value
            return obj

    class BaseEvent:  # type: ignore
        def __init__(self, event_type: Any):
            self.event_type = event_type

logger = logging.getLogger(__name__)


class EventBus:
    """Asyncio event bus for EMP system communication."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, Set[Callable]] = defaultdict(set)
        self._event_history: List[BaseEvent] = []
        self._max_history: int = 10000
        self._running: bool = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        
        logger.info("Event Bus initialized")
        
    async def start(self):
        """Start the event bus."""
        if self._running:
            logger.warning("Event Bus already running")
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event Bus started")
        
    async def stop(self):
        """Stop the event bus."""
        if not self._running:
            logger.warning("Event Bus not running")
            return
            
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        logger.info("Event Bus stopped")
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to an event type."""
        self._subscribers[event_type].add(callback)
        logger.debug(f"Subscribed to {event_type.value}: {callback.__name__}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)
            logger.debug(f"Unsubscribed from {event_type.value}: {callback.__name__}")
            
    async def publish(self, event: BaseEvent):
        """Publish an event to the bus."""
        if not self._running:
            logger.warning("Event Bus not running, cannot publish event")
            return
            
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
            
        # Add to processing queue
        await self._queue.put(event)
        logger.debug(f"Published event: {event.event_type.value}")
        
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                # Get subscribers for this event type
                subscribers = self._subscribers.get(event.event_type, set())
                
                if not subscribers:
                    logger.debug(f"No subscribers for event type: {event.event_type.value}")
                    continue
                    
                # Create tasks for all subscribers
                tasks = []
                for callback in subscribers:
                    task = asyncio.create_task(self._call_subscriber(callback, event))
                    tasks.append(task)
                    
                # Wait for all subscribers to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                
    async def _call_subscriber(self, callback: Callable, event: BaseEvent):
        """Call a subscriber with an event."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            logger.error(f"Error in subscriber {callback.__name__}: {e}")
            
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         limit: Optional[int] = None) -> List[BaseEvent]:
        """Get event history, optionally filtered by type."""
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
            
        if limit:
            history = history[-limit:]
            
        return history
        
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get the number of subscribers for an event type."""
        return len(self._subscribers.get(event_type, set()))
        
    def get_all_subscribers(self) -> Dict[EventType, Set[Callable]]:
        """Get all subscribers."""
        return dict(self._subscribers)
        
    def clear_history(self):
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")
        
    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running
        
    def get_queue_size(self) -> int:
        """Get the current queue size."""
        return self._queue.qsize()


# Global event bus instance
event_bus = EventBus()


# Convenience functions for global event bus
async def publish_event(event: BaseEvent):
    """Publish an event to the global event bus."""
    await event_bus.publish(event)


def subscribe_to_event(event_type: EventType, callback: Callable):
    """Subscribe to an event type on the global event bus."""
    event_bus.subscribe(event_type, callback)


def unsubscribe_from_event(event_type: EventType, callback: Callable):
    """Unsubscribe from an event type on the global event bus."""
    event_bus.unsubscribe(event_type, callback)


async def start_event_bus():
    """Start the global event bus."""
    await event_bus.start()


async def stop_event_bus():
    """Stop the global event bus."""
    await event_bus.stop()

# -------------------------
# Minimal in-process topic bus (additive to existing asyncio EventBus)
# -------------------------

@dataclass(frozen=True)
class Subscription:
    topic: str
    handler: Callable[[Any], Any]
    id: int


class EventBusV2(EventBus):
    """Minimal in-process pub/sub (topic-based), added without breaking existing API."""
    def __init__(self) -> None:
        super().__init__()
        self._topic_handlers: Dict[str, Set[Callable]] = defaultdict(set)
        self._lock = threading.Lock()
        self._next_id: int = 1
        self._subs_by_id: Dict[int, Subscription] = {}

    def subscribe(self, topic: str, handler: Callable) -> Subscription:
        """Subscribe a handler to a string topic. Returns a Subscription handle."""
        if not isinstance(topic, str):
            raise TypeError("topic must be a str")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._topic_handlers[topic].add(handler)
            sub = Subscription(topic=topic, handler=handler, id=self._next_id)
            self._subs_by_id[self._next_id] = sub
            self._next_id += 1
            return sub

    def unsubscribe(self, handle) -> None:
        """Unsubscribe via Subscription handle (idempotent). Also accepts tuple(topic, handler)."""
        topic = None
        handler = None
        sub_id = None

        if isinstance(handle, Subscription):
            topic = handle.topic
            handler = handle.handler
            sub_id = handle.id
        elif isinstance(handle, tuple) and len(handle) == 2 and isinstance(handle[0], str) and callable(handle[1]):
            topic, handler = handle  # type: ignore[assignment]
        else:
            if isinstance(handle, int):
                sub_id = handle

        with self._lock:
            if sub_id is not None and sub_id in self._subs_by_id:
                sub = self._subs_by_id.pop(sub_id, None)
                if sub is not None:
                    topic = sub.topic
                    handler = sub.handler

            if topic is not None and handler is not None:
                handlers = self._topic_handlers.get(topic)
                if handlers is not None:
                    handlers.discard(handler)
                    if not handlers:
                        self._topic_handlers.pop(topic, None)

    def publish(self, topic: str, payload: Any) -> int:
        """Publish payload to topic; returns number of handlers invoked."""
        with self._lock:
            handlers = list(self._topic_handlers.get(topic, set()))
        invoked = 0
        for fn in handlers:
            try:
                fn(payload)
                invoked += 1
            except Exception:
                logger.exception("Error in handler for topic %s", topic)
        return invoked


# Module-level singleton accessor for the minimal (topic-based) bus
_GLOBAL_TOPIC_BUS: Optional[EventBus] = None

def get_global_bus() -> EventBus:
    """Return singleton instance of minimal in-process EventBus (topic-based)."""
    global _GLOBAL_TOPIC_BUS
    if _GLOBAL_TOPIC_BUS is None:
        _GLOBAL_TOPIC_BUS = EventBusV2()
    return _GLOBAL_TOPIC_BUS
