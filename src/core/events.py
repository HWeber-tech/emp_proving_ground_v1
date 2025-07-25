#!/usr/bin/env python3
"""
Core Events - Epic 4: Fusing Foresight
Pydantic models for events including the new MarketForecast integration.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EventType(str, Enum):
    """Event types for the system."""
    MARKET_UNDERSTANDING = "market_understanding"
    CONTEXT_UPDATE = "context_update"
    DECISION_MADE = "decision_made"
    TRADE_EXECUTED = "trade_executed"

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
