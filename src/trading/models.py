"""
Trading Models
try:
    from src.core.market_data import MarketData  # legacy
except Exception:  # pragma: no cover
    MarketData = object  # type: ignore
Core trading models for positions, signals, and market data
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"

@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    symbol: str
    signal_type: SignalType
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    confidence: float = 0.5
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

from trading.models.position import Position as Position

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot"""
    total_value: float
    cash_balance: float
    positions: List[Position]
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def position_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.value for pos in self.positions)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L"""
        return self.unrealized_pnl + self.realized_pnl
