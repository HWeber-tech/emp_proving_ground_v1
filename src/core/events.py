"""
EMP Core Events v1.1

Defines the core event models for inter-layer communication
in the EMP Ultimate Architecture v1.1.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np


class EventType(Enum):
    """Types of events in the EMP system."""
    MARKET_UNDERSTANDING = "market_understanding"
    CONTEXT_PACKET = "context_packet"
    TRADE_INTENT = "trade_intent"
    FITNESS_REPORT = "fitness_report"
    SENSORY_SIGNAL = "sensory_signal"
    THINKING_ANALYSIS = "thinking_analysis"
    EVOLUTION_EVENT = "evolution_event"
    GOVERNANCE_DECISION = "governance_decision"
    OPERATIONAL_STATUS = "operational_status"


class MarketData(BaseModel):
    """Market data structure for sensory processing."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "unknown"
    latency_ms: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist()
        }


class SensorySignal(BaseModel):
    """Sensory signal from sensory organs."""
    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketUnderstanding(BaseModel):
    """Market understanding from sensory layer."""
    timestamp: datetime
    symbol: str
    signals: List[SensorySignal] = Field(default_factory=list)
    composite_score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('composite_score')
    def validate_composite_score(cls, v):
        return max(-1.0, min(1.0, v))
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisResult(BaseModel):
    """Analysis result from thinking layer."""
    timestamp: datetime
    analysis_type: str
    result: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextPacket(BaseModel):
    """Context packet from thinking layer."""
    timestamp: datetime
    market_understanding: MarketUnderstanding
    analyses: List[AnalysisResult] = Field(default_factory=list)
    context_score: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('context_score')
    def validate_context_score(cls, v):
        return max(-1.0, min(1.0, v))
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TradeIntent(BaseModel):
    """Trade intent from adaptive core."""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    strategy_id: str
    genome_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('action')
    def validate_action(cls, v):
        if v not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError('Action must be BUY, SELL, or HOLD')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics for fitness evaluation."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskMetrics(BaseModel):
    """Risk metrics for fitness evaluation."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    current_drawdown: float = 0.0
    risk_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FitnessReport(BaseModel):
    """Fitness report from simulation envelope."""
    timestamp: datetime
    genome_id: str
    strategy_id: str
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    fitness_score: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('fitness_score')
    def validate_fitness_score(cls, v):
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EvolutionEvent(BaseModel):
    """Evolution event from adaptive core."""
    timestamp: datetime
    event_type: str  # 'generation_complete', 'mutation', 'crossover', 'selection'
    genome_id: str
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GovernanceDecision(BaseModel):
    """Governance decision from governance layer."""
    timestamp: datetime
    decision_type: str  # 'approve', 'reject', 'escalate', 'auto_approve'
    strategy_id: str
    genome_id: str
    approver: Optional[str] = None
    reason: Optional[str] = None
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OperationalStatus(BaseModel):
    """Operational status from operational backbone."""
    timestamp: datetime
    component: str
    status: str  # 'healthy', 'warning', 'critical', 'offline'
    metrics: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BaseEvent(BaseModel):
    """Base event class for all EMP events."""
    event_type: EventType
    timestamp: datetime
    source: str
    target: Optional[str] = None
    payload: Union[
        MarketUnderstanding,
        ContextPacket,
        TradeIntent,
        FitnessReport,
        SensorySignal,
        AnalysisResult,
        EvolutionEvent,
        GovernanceDecision,
        OperationalStatus
    ]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True 