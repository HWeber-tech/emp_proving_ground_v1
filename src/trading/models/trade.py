"""
Trade Models
============

Data models for executed trades in the EMP Proving Ground system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional


@dataclass
class Trade:
    """Represents an executed trade."""

    trade_id: str
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]  # constrained literal
    quantity: float
    price: float
    timestamp: datetime | str
    commission: float = 0.0
    exchange: str | None = None

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
