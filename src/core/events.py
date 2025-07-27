#!/usr/bin/env python3
"""
Core Events - Epic 4: Fusing Foresight
Pydantic models for events including the new MarketForecast integration.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class EventType(str, Enum):
    """Event types for the system."""
    MARKET_UNDERSTANDING = "market_understanding"
    CONTEXT_UPDATE = "context_update"
    DECISION_MADE = "decision_made"
    TRADE_EXECUTED = "trade_executed"
    TRADE_INTENT = "trade_intent"
    EXECUTION_REPORT = "execution_report"
    FITNESS_REPORT = "fitness_report"
    GENOME_UPDATE = "genome_update"
    TELEMETRY = "telemetry"

class BaseEvent(BaseModel):
    """Base event class."""
    event_type: EventType
    timestamp: str
    sequence_id: str

class MarketForecast(BaseModel):
    """Market forecast from the predictive model."""
    prob_up: float
    prob_down: float
    prob_flat: float
    model_version: str  # The MLflow run ID
    timestamp: str

class PerformanceMetrics(BaseModel):
    """Performance metrics for trading and system evaluation."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    timestamp: str

class RiskMetrics(BaseModel):
    """Risk metrics for portfolio and position management."""
    portfolio_value: float
    exposure: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    beta: float
    volatility: float
    correlation: float
    timestamp: str

class ContextPacket(BaseModel):
    """Enhanced context packet with market forecast."""
    # Existing fields
    timestamp: str
    market_data: Dict[str, Any]
    risk_metrics: Dict[str, float]
    
    # New field for predictive intelligence
    market_forecast: Optional[MarketForecast] = None
    
    # Additional metadata
    sequence_id: str
    source: str = "thinking_manager"

class MarketUnderstanding(BaseModel):
    """Market understanding event."""
    timestamp: str
    market_state: str
    confidence: float
    reasoning: str

class TradeIntent(BaseModel):
    """Trade intent event."""
    timestamp: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: float
    price: float
    confidence: float

class ExecutionReport(BaseModel):
    """Execution report event."""
    timestamp: str
    order_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    status: str  # FILLED, PARTIAL, REJECTED, CANCELLED

class FitnessReport(BaseModel):
    """Fitness report event."""
    timestamp: str
    strategy_id: str
    fitness_score: float
    metrics: Dict[str, float]

class GenomeUpdate(BaseModel):
    """Genome update event."""
    timestamp: str
    genome_id: str
    changes: Dict[str, Any]
    reason: str

class Telemetry(BaseModel):
    """Telemetry event."""
    timestamp: str
    component: str
    metric: str
    value: float
    metadata: Dict[str, Any] = {}

class OrderBookLevel(BaseModel):
    """Order book level."""
    price: float
    size: float
    count: int = 1

class OrderBook(BaseModel):
    """Order book representation."""
    symbol: str
    timestamp: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float

class EventBus:
    """Simple event bus for system communication."""
    def __init__(self):
        self.handlers = {}
    
    def subscribe(self, event_type, handler):
        """Subscribe to events of a specific type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def publish(self, event):
        """Publish an event to all subscribers."""
        event_type = event.event_type
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)
    
    async def start(self):
        """Start the event bus."""
        pass
    
    async def stop(self):
        """Stop the event bus."""
        pass
