"""
Event Bus - Simple Implementation
================================

Provides event-driven communication for Phase 3 systems.
"""

import asyncio
import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for system communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history = []
        
    async def subscribe(self, event_type: str, callback: Callable) -> bool:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        return True
    
    async def emit(self, event_type: str, data: Any) -> bool:
        """Emit an event."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self._event_history.append(event)
        
        # Keep only last 1000 events
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-1000:]
        
        # Notify subscribers
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        return True
    
    async def get_events(self, event_type: Optional[str] = None, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events."""
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        return events[-limit:]
    
    async def clear_events(self) -> bool:
        """Clear event history."""
        self._event_history.clear()
        return True


# Global instance
_event_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get or create global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Event models for Phase 3
class AlgorithmSignature:
    """Represents an algorithmic trading signature."""
    
    def __init__(self, signature_id: str, pattern: str, confidence: float):
        self.signature_id = signature_id
        self.pattern = pattern
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signature_id': self.signature_id,
            'pattern': self.pattern,
            'confidence': self.confidence
        }


class CompetitorBehavior:
    """Represents competitor behavior analysis."""
    
    def __init__(self, competitor_id: str, behavior_type: str, 
                 frequency: float, impact: float):
        self.competitor_id = competitor_id
        self.behavior_type = behavior_type
        self.frequency = frequency
        self.impact = impact
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'competitor_id': self.competitor_id,
            'behavior_type': self.behavior_type,
            'frequency': self.frequency,
            'impact': self.impact
        }


class CounterStrategy:
    """Represents a counter-strategy."""
    
    def __init__(self, strategy_id: str, target_behavior: str, 
                 effectiveness: float, parameters: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.target_behavior = target_behavior
        self.effectiveness = effectiveness
        self.parameters = parameters
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'target_behavior': self.target_behavior,
            'effectiveness': self.effectiveness,
            'parameters': self.parameters
        }
