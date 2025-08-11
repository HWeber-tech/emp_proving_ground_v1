"""
EMP Genome Model v1.1

Defines the genetic encoding structure for trading strategies,
risk parameters, and timing preferences in the adaptive core.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.core.exceptions import GenomeException

logger = logging.getLogger(__name__)


class GenomeType(Enum):
    """Types of genome components."""
    STRATEGY = "strategy"
    RISK = "risk"
    TIMING = "timing"
    SENSORY = "sensory"
    THINKING = "thinking"


@dataclass
class StrategyGenome:
    """Strategy-related genetic parameters."""
    strategy_type: str
    entry_threshold: float
    exit_threshold: float
    momentum_weight: float
    trend_weight: float
    volume_weight: float
    sentiment_weight: float
    lookback_period: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskGenome:
    """Risk-related genetic parameters."""
    risk_tolerance: float
    position_size_multiplier: float
    stop_loss_threshold: float
    take_profit_threshold: float
    max_drawdown_limit: float
    volatility_threshold: float
    correlation_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingGenome:
    """Timing-related genetic parameters."""
    entry_timing: str  # 'immediate', 'confirmation', 'pullback'
    exit_timing: str   # 'immediate', 'trailing', 'time_based'
    holding_period_min: int
    holding_period_max: int
    reentry_delay: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensoryGenome:
    """Sensory-related genetic parameters."""
    price_weight: float
    volume_weight: float
    orderbook_weight: float
    news_weight: float
    sentiment_weight: float
    economic_weight: float
    signal_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkingGenome:
    """Thinking-related genetic parameters."""
    trend_analysis_weight: float
    risk_analysis_weight: float
    performance_analysis_weight: float
    pattern_recognition_weight: float
    inference_confidence_threshold: float
    analysis_methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Genome:
    """Composite genome container (placeholder to fix parser error)."""
    strategy: Optional[StrategyGenome] = None
    risk: Optional[RiskGenome] = None
    timing: Optional[TimingGenome] = None
    sensory: Optional[SensoryGenome] = None
    thinking: Optional[ThinkingGenome] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
