"""
Core Event Contracts for EMP Ultimate Architecture v1.1
Defines all inter-layer communication contracts using Pydantic models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Standardized event types for the EMP system."""
    MARKET_UNDERSTANDING = "market_understanding"
    CONTEXT_PACKET = "context_packet"
    TRADE_INTENT = "trade_intent"
    EXECUTION_REPORT = "execution_report"
    FITNESS_REPORT = "fitness_report"
    GENOME_UPDATE = "genome_update"
    TELEMETRY = "telemetry"


class BaseEvent(BaseModel):
    """Base class for all events in the EMP system."""
    event_id: str
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics for strategies and components."""
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_return: Decimal
    volatility: Decimal
    trades: int
    avg_trade_duration: int


class MarketUnderstanding(BaseEvent):
    """Event published by Sensory Layer to Thinking Layer."""
    symbol: str
    price: Decimal
    volume: Decimal
    indicators: Dict[str, Decimal] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextPacket(BaseEvent):
    """Event published by Thinking Layer to Adaptive Core."""
    regime: str
    patterns: Dict[str, Any] = Field(default_factory=dict)
    risk_metrics: Dict[str, Decimal] = Field(default_factory=dict)
    confidence: Decimal
    latent_vec: List[float] = Field(default_factory=list)  # Vector for pattern memory
    market_state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TradeIntent(BaseEvent):
    """Event published by Adaptive Core to Trading Layer."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: str = "MARKET"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionReport(BaseEvent):
    """Event published by Trading Layer after trade execution."""
    trade_intent_id: str
    symbol: str
    action: str
    quantity: Decimal
    price: Decimal
    fees: Decimal = Decimal('0')
    status: str  # FILLED, PARTIAL, REJECTED, ERROR
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FitnessReport(BaseEvent):
    """Event published by Simulation Envelope to Adaptive Core."""
    genome_id: str
    fitness_score: Decimal
    metrics: Dict[str, Decimal] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenomeUpdate(BaseEvent):
    """Event for genome evolution notifications."""
    genome_id: str
    action: str  # CREATED, UPDATED, PROMOTED, RETIRED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Telemetry(BaseEvent):
    """System health and performance telemetry."""
    component: str
    metric: str
    value: Decimal
    tags: Dict[str, str] = Field(default_factory=dict)


class SensorySignal(BaseEvent):
    """Signal from sensory organs."""
    signal_type: str
    value: Decimal
    confidence: Decimal
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseEvent):
    """Result from thinking layer analysis."""
    analysis_type: str
    result: Dict[str, Any]
    confidence: Decimal
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GovernanceDecision(BaseEvent):
    """Decision from governance layer."""
    decision_type: str
    approved: bool
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarketData(BaseModel):
    """Market data structure for sensory input."""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None


class RiskMetrics(BaseModel):
    """Risk metrics for trading and governance."""
    var_95: Decimal
    var_99: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    beta: Decimal
    alpha: Decimal


class EvolutionEvent(BaseEvent):
    """Events related to genetic evolution."""
    generation: int
    population_size: int
    best_fitness: Decimal
    average_fitness: Decimal
    diversity_score: Decimal
    metadata: Dict[str, Any] = Field(default_factory=dict)
