"""
EMP Core Events v1.1

Defines the event types and structures used throughout the EMP system
for inter-layer communication and system coordination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class EventType(Enum):
    """Event types for system-wide event classification."""
    
    # Sensory events
    SENSORY_SIGNAL_RECEIVED = "sensory.signal.received"
    SENSORY_ORGAN_CALIBRATED = "sensory.organ.calibrated"
    
    # Thinking events
    ANALYSIS_COMPLETED = "thinking.analysis.completed"
    PATTERN_DETECTED = "thinking.pattern.detected"
    INFERENCE_MADE = "thinking.inference.made"
    
    # Trading events
    TRADING_DECISION_MADE = "trading.decision.made"
    ORDER_EXECUTED = "trading.order.executed"
    POSITION_OPENED = "trading.position.opened"
    POSITION_CLOSED = "trading.position.closed"
    
    # Evolution events
    GENERATION_COMPLETED = "evolution.generation.completed"
    FITNESS_EVALUATED = "evolution.fitness.evaluated"
    POPULATION_EVOLVED = "evolution.population.evolved"
    
    # Governance events
    STRATEGY_APPROVED = "governance.strategy.approved"
    STRATEGY_REJECTED = "governance.strategy.rejected"
    AUDIT_LOG_CREATED = "governance.audit.log.created"
    
    # Operational events
    SYSTEM_STARTED = "operational.system.started"
    SYSTEM_STOPPED = "operational.system.stopped"
    HEALTH_CHECK_FAILED = "operational.health.check.failed"


@dataclass
class BaseEvent:
    """Base event class with common properties."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""


@dataclass
class SensoryEvent(BaseEvent):
    """Sensory layer events."""
    signal: Optional[Dict[str, Any]] = None
    organ_id: str = ""


@dataclass
class ThinkingEvent(BaseEvent):
    """Thinking layer events."""
    analysis_result: Optional[Dict[str, Any]] = None
    pattern_id: str = ""


@dataclass
class TradingEvent(BaseEvent):
    """Trading layer events."""
    decision: Optional[Dict[str, Any]] = None
    strategy_id: str = ""
    symbol: str = ""


@dataclass
class EvolutionEvent(BaseEvent):
    """Evolution layer events."""
    population_size: int = 0
    generation: int = 0
    best_fitness: float = 0.0


@dataclass
class GovernanceEvent(BaseEvent):
    """Governance layer events."""
    strategy_id: str = ""
    approval_status: str = ""
    reason: str = ""


@dataclass
class OperationalEvent(BaseEvent):
    """Operational layer events."""
    component: str = ""
    status: str = ""
    error_message: str = ""


# Event factory functions
def create_sensory_event(signal: Dict[str, Any], organ_id: str) -> SensoryEvent:
    """Create a sensory event."""
    return SensoryEvent(
        event_type=EventType.SENSORY_SIGNAL_RECEIVED,
        signal=signal,
        organ_id=organ_id
    )


def create_thinking_event(analysis_result: Dict[str, Any], pattern_id: str) -> ThinkingEvent:
    """Create a thinking event."""
    return ThinkingEvent(
        event_type=EventType.ANALYSIS_COMPLETED,
        analysis_result=analysis_result,
        pattern_id=pattern_id
    )


def create_trading_event(decision: Dict[str, Any], strategy_id: str, symbol: str) -> TradingEvent:
    """Create a trading event."""
    return TradingEvent(
        event_type=EventType.TRADING_DECISION_MADE,
        decision=decision,
        strategy_id=strategy_id,
        symbol=symbol
    )


def create_evolution_event(population_size: int, generation: int, best_fitness: float) -> EvolutionEvent:
    """Create an evolution event."""
    return EvolutionEvent(
        event_type=EventType.GENERATION_COMPLETED,
        population_size=population_size,
        generation=generation,
        best_fitness=best_fitness
    ) 