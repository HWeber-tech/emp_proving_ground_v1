"""
Core Event Models for EMP v4.0
Defines the primary sensory events and data structures
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
from enum import Enum


class OrderBookLevel(BaseModel):
    """Represents a single price level in the order book."""
    price: float
    size: float  # Volume at this price level
    
    def __str__(self):
        return f"Level(price={self.price}, size={self.size})"


class OrderBook(BaseModel):
    """Represents a snapshot of the full order book."""
    bids: List[OrderBookLevel] = Field(default_factory=list)
    asks: List[OrderBookLevel] = Field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def add_bid(self, price: float, size: float):
        """Add a bid level, maintaining sorted order (highest first)."""
        self.bids.append(OrderBookLevel(price=price, size=size))
        self.bids.sort(key=lambda x: x.price, reverse=True)
    
    def add_ask(self, price: float, size: float):
        """Add an ask level, maintaining sorted order (lowest first)."""
        self.asks.append(OrderBookLevel(price=price, size=size))
        self.asks.sort(key=lambda x: x.price)


class MarketUnderstanding(BaseModel):
    """
    The primary sensory event of the system.
    v4.0 Schema: Enriched with full order book depth.
    """
    timestamp: datetime
    symbol: str
    
    # Top-of-book (for legacy/simple strategies)
    best_bid: float
    best_ask: float
    last_trade_price: Optional[float] = None
    
    # The new, high-resolution data
    order_book: OrderBook  # The full depth of market
    
    # Metadata for tracking
    source: str = "FIX"  # Source of this market data
    sequence_number: Optional[int] = None  # FIX sequence number
    
    # Derived metrics (can be calculated later)
    spread: Optional[float] = None
    mid_price: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.order_book:
            self.spread = self.order_book.spread
            if self.order_book.best_bid and self.order_book.best_ask:
                self.mid_price = (self.order_book.best_bid + self.order_book.best_ask) / 2
    
    def __str__(self):
        return f"MarketUnderstanding({self.symbol} @ {self.timestamp})"


class MarketDataRequest(BaseModel):
    """Request for market data subscription."""
    symbol: str
    depth: int = 0  # 0 for full depth
    subscription_type: str = "SNAPSHOT_PLUS_UPDATES"


class MarketDataSubscription(BaseModel):
    """Active market data subscription."""
    symbol: str
    subscription_id: str
    depth: int
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OrderStatus(str, Enum):
    """Order status enumeration for trade execution"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    DONE_FOR_DAY = "DONE_FOR_DAY"
    CANCELED = "CANCELED"
    REPLACED = "REPLACED"
    PENDING_CANCEL = "PENDING_CANCEL"
    STOPPED = "STOPPED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ExecutionReportEvent(BaseModel):
    """Execution report event for trade confirmations"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    price: Decimal
    filled_price: Decimal
    status: OrderStatus
    timestamp: datetime
    exec_id: Optional[str] = None
    account: Optional[str] = None
    text: Optional[str] = None
    
    def __str__(self):
        return f"ExecutionReport({self.order_id}: {self.status} {self.side} {self.symbol} {self.filled_quantity}/{self.quantity}@{self.filled_price})"


class TradeIntent(BaseModel):
    """Trade intent for order placement"""
    symbol: str
    side: str  # BUY or SELL
    quantity: Decimal
    order_type: str  # 1=Market, 2=Limit, 4=Stop
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "0"  # 0=Day, 1=GTC, 3=IOC, 4=FOK
    account: Optional[str] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __str__(self):
        return f"TradeIntent({self.side} {self.quantity} {self.symbol} {self.order_type})"
