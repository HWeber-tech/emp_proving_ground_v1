"""
Order Models
============

Data models for trading orders in the EMP Proving Ground system.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Order:
    """Represents a trading order."""
    
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: str
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (pending or partially filled)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    def update_status(self, new_status: OrderStatus) -> None:
        """Update order status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.now()
    
    def add_fill(self, quantity: float, price: float) -> None:
        """Add a fill to the order."""
        self.filled_quantity += quantity
        if self.average_price is None:
            self.average_price = price
        else:
            # Calculate weighted average price
            total_value = (self.average_price * (self.filled_quantity - quantity)) + (price * quantity)
            self.average_price = total_value / self.filled_quantity
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now()
