"""
Canonical order book models for trading domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class OrderBookLevel:
    price: float
    volume: float
    orders: int = 1
    timestamp: Optional[datetime] = None


@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    spread: float = 0.0
    mid_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    total_bid_volume: float = 0.0
    total_ask_volume: float = 0.0

    def compute(self) -> "OrderBookSnapshot":
        if self.bids and self.asks:
            self.best_bid = self.bids[0].price
            self.best_ask = self.asks[0].price
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2.0
        else:
            self.best_bid = 0.0
            self.best_ask = 0.0
            self.spread = 0.0
            self.mid_price = 0.0
        self.total_bid_volume = sum(l.volume for l in self.bids)
        self.total_ask_volume = sum(l.volume for l in self.asks)
        return self

    def __post_init__(self) -> None:
        # auto compute derived metrics when created
        self.compute()


__all__ = ["OrderBookLevel", "OrderBookSnapshot"]