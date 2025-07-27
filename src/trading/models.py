"""
Trading Models
from src.core.market_data import MarketData
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

@dataclass
class Position:
    """Trading position"""
    position_id: str
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    current_price: float
    status: PositionStatus = PositionStatus.OPEN
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
    
    @property
    def value(self) -> float:
        """Current position value"""
        return abs(self.size * self.current_price)
    
    @property
    def pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_price(self, new_price: float) -> None:
        """Update current price and unrealized P&L"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.size
    
    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """Close the position"""
        self.current_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.realized_pnl = (exit_price - self.entry_price) * self.size
        self.unrealized_pnl = 0.0
        self.status = PositionStatus.CLOSED

@dataclass
class MarketData:
    """Market data snapshot"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    
    def __post_init__(self):
        if self.spread is None and self.bid and self.ask:
            self.spread = self.ask - self.bid

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
