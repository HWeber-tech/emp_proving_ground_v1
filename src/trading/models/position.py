"""
Position Models
===============

Data models for trading positions in the EMP Proving Ground system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Position:
    """Represents a trading position."""
    
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_price: Optional[float] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        if self.market_price is None:
            return 0.0
        return self.quantity * self.market_price
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0
    
    def update_market_price(self, new_price: float) -> None:
        """Update market price and recalculate unrealized P&L."""
        self.market_price = new_price
        if self.quantity != 0:
            self.unrealized_pnl = (new_price - self.average_price) * self.quantity
        self.last_updated = datetime.now()
    
    def add_realized_pnl(self, pnl: float) -> None:
        """Add realized P&L to the position."""
        self.realized_pnl += pnl
        self.last_updated = datetime.now()
    
    def update_quantity(self, new_quantity: float, new_average_price: Optional[float] = None) -> None:
        """Update position quantity and optionally average price."""
        self.quantity = new_quantity
        if new_average_price is not None:
            self.average_price = new_average_price
        self.last_updated = datetime.now()
