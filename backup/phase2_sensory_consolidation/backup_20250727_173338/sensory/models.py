"""
Sensory Models
Models for sensory processing and technical analysis
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum


class Sentiment(Enum):
    """Market sentiment"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TechnicalSignalType(Enum):
    """Types of technical signals"""
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    MACD_BULLISH_CROSS = "macd_bullish_cross"
    MACD_BEARISH_CROSS = "macd_bearish_cross"
    BB_UPPER_TOUCH = "bb_upper_touch"
    BB_LOWER_TOUCH = "bb_lower_touch"
    VOLUME_SPIKE = "volume_spike"
    MOMENTUM_BULLISH = "momentum_bullish"
    MOMENTUM_BEARISH = "momentum_bearish"
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"


@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    signal_type: TechnicalSignalType
    value: float
    threshold: float
    confidence: float
    timeframe: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SensoryReading:
    """Complete sensory reading"""
    symbol: str
    timeframe: str
    timestamp: datetime
    overall_sentiment: str
    confidence_score: float
    technical_signals: List[TechnicalSignal]
    market_context: Dict
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketContext:
    """Market context information"""
    symbol: str
    timeframe: str
    current_price: float
    price_change_24h: float
    volume_24h: float
    volatility: float
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
