"""
Unified Market Data Structure
Consolidates all MarketData implementations into a single, canonical version
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal


@dataclass
class MarketData:
    """
    Unified market data structure that consolidates all previous implementations.
    
    This replaces the 8+ different MarketData classes found across the codebase
    with a single, comprehensive structure that supports all use cases.
    """
    
    # Core identifiers
    symbol: str
    timestamp: datetime
    
    # Price data
    bid: Decimal
    ask: Decimal
    last: Decimal
    high: Decimal
    low: Decimal
    open: Decimal
    close: Decimal
    
    # Volume data
    volume: Decimal
    bid_volume: Optional[Decimal] = None
    ask_volume: Optional[Decimal] = None
    
    # Market depth (Level 2)
    bids: Optional[Dict[Decimal, Decimal]] = None  # price -> volume
    asks: Optional[Dict[Decimal, Decimal]] = None  # price -> volume
    
    # Additional market data
    spread: Optional[Decimal] = None
    mid_price: Optional[Decimal] = None
    
    # Exchange/broker specific
    exchange: Optional[str] = None
    source: Optional[str] = None
    
    # Quality indicators
    is_real: bool = True  # True for real data, False for mock/simulated
    latency_ms: Optional[int] = None
    
    # Metadata
    sequence_number: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate derived values after initialization"""
        if self.spread is None and self.bid and self.ask:
            self.spread = self.ask - self.bid
            
        if self.mid_price is None and self.bid and self.ask:
            self.mid_price = (self.bid + self.ask) / 2
    
    @property
    def price(self) -> Decimal:
        """Alias for last price for backward compatibility"""
        return self.last
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create MarketData from dictionary (for backward compatibility)"""
        # Handle various legacy formats
        symbol = data.get('symbol', data.get('instrument', 'UNKNOWN'))
        
        # Handle different timestamp formats
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        elif timestamp is None:
            timestamp = datetime.utcnow()
            
        # Handle decimal conversion
        def to_decimal(value, default=Decimal('0')):
            if value is None:
                return default
            return Decimal(str(value))
            
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            bid=to_decimal(data.get('bid', data.get('Bid', 0))),
            ask=to_decimal(data.get('ask', data.get('Ask', 0))),
            last=to_decimal(data.get('last', data.get('Last', 0))),
            high=to_decimal(data.get('high', data.get('High', 0))),
            low=to_decimal(data.get('low', data.get('Low', 0))),
            open=to_decimal(data.get('open', data.get('Open', 0))),
            close=to_decimal(data.get('close', data.get('Close', 0))),
            volume=to_decimal(data.get('volume', data.get('Volume', 0))),
            bid_volume=to_decimal(data.get('bid_volume')) if 'bid_volume' in data else None,
            ask_volume=to_decimal(data.get('ask_volume')) if 'ask_volume' in data else None,
            exchange=data.get('exchange'),
            source=data.get('source', 'unknown'),
            is_real=data.get('is_real', True),
            latency_ms=data.get('latency_ms'),
            raw_data=data if data.get('raw_data') is None else data.get('raw_data')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bid': str(self.bid),
            'ask': str(self.ask),
            'last': str(self.last),
            'high': str(self.high),
            'low': str(self.low),
            'open': str(self.open),
            'close': str(self.close),
            'volume': str(self.volume),
            'spread': str(self.spread) if self.spread else None,
            'mid_price': str(self.mid_price) if self.mid_price else None,
            'exchange': self.exchange,
            'source': self.source,
            'is_real': self.is_real,
            'latency_ms': self.latency_ms
        }
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return f"MarketData({self.symbol}, {self.timestamp}, bid={self.bid}, ask={self.ask}, last={self.last})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Backward compatibility aliases for smooth migration
MarketDataSnapshot = MarketData
MarketDataStructure = MarketData
MarketDataEntry = MarketData
