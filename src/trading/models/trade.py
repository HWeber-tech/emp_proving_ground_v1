"""
Trade Models
============

Data models for executed trades in the EMP Proving Ground system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Represents an executed trade."""
    
    trade_id: str
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime | str
    commission: float = 0.0
    exchange: Optional[str] = None
    
    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """Calculate net trade value after commission."""
        return self.value - self.commission
