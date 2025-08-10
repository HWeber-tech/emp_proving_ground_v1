"""
Unit tests for the consolidated MarketData class.
Tests the unified market data structure functionality.
"""

import pytest
from datetime import datetime
from decimal import Decimal
try:
    from src.core.market_data import MarketData  # legacy
except Exception:  # pragma: no cover
    MarketData = None  # type: ignore


class TestMarketData:
    """Test suite for MarketData functionality."""
    
    def test_basic_creation(self):
        """Test basic MarketData creation."""
        timestamp = datetime.now()
        
        data = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.0950"),
            ask=Decimal("1.0952"),
            last=Decimal("1.0951"),
            high=Decimal("1.0960"),
            low=Decimal("1.0940"),
            open=Decimal("1.0945"),
            close=Decimal("1.0951"),
            volume=Decimal("1000000")
        )
        
        assert data.symbol == "EURUSD"
        assert data.timestamp == timestamp
        assert data.bid == Decimal("1.0950")
        assert data.ask == Decimal("1.0952")
        assert data.volume == Decimal("1000000")
        
    def test_optional_fields(self):
        """Test MarketData with optional fields."""
        timestamp = datetime.now()
        
        data = MarketData(
            symbol="GBPUSD",
            timestamp=timestamp,
            bid=Decimal("1.2500"),
            ask=Decimal("1.2502"),
            last=Decimal("1.2501"),
            high=Decimal("1.2510"),
            low=Decimal("1.2490"),
            open=Decimal("1.2495"),
            close=Decimal("1.2501"),
            volume=Decimal("500000"),
            bid_volume=Decimal("250000"),
            ask_volume=Decimal("250000"),
            spread=Decimal("0.0002"),
            mid_price=Decimal("1.2501"),
            exchange="ICMarkets",
            source="FIX_API"
        )
        
        assert data.bid_volume == Decimal("250000")
        assert data.ask_volume == Decimal("250000")
        assert data.spread == Decimal("0.0002")
        assert data.mid_price == Decimal("1.2501")
        assert data.exchange == "ICMarkets"
        assert data.source == "FIX_API"
        
    def test_market_depth(self):
        """Test MarketData with market depth (Level 2) data."""
        timestamp = datetime.now()
        
        bids = {
            Decimal("1.0950"): Decimal("100000"),
            Decimal("1.0949"): Decimal("200000"),
            Decimal("1.0948"): Decimal("150000")
        }
        
        asks = {
            Decimal("1.0952"): Decimal("100000"),
            Decimal("1.0953"): Decimal("200000"),
            Decimal("1.0954"): Decimal("150000")
        }
        
        data = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.0950"),
            ask=Decimal("1.0952"),
            last=Decimal("1.0951"),
            high=Decimal("1.0960"),
            low=Decimal("1.0940"),
            open=Decimal("1.0945"),
            close=Decimal("1.0951"),
            volume=Decimal("1000000"),
            bids=bids,
            asks=asks
        )
        
        assert data.bids == bids
        assert data.asks == asks
        assert len(data.bids) == 3
        assert len(data.asks) == 3
        
    def test_dataclass_equality(self):
        """Test MarketData equality comparison."""
        timestamp = datetime.now()
        
        data1 = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.0950"),
            ask=Decimal("1.0952"),
            last=Decimal("1.0951"),
            high=Decimal("1.0960"),
            low=Decimal("1.0940"),
            open=Decimal("1.0945"),
            close=Decimal("1.0951"),
            volume=Decimal("1000000")
        )
        
        data2 = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.0950"),
            ask=Decimal("1.0952"),
            last=Decimal("1.0951"),
            high=Decimal("1.0960"),
            low=Decimal("1.0940"),
            open=Decimal("1.0945"),
            close=Decimal("1.0951"),
            volume=Decimal("1000000")
        )
        
        assert data1 == data2
        
    def test_dataclass_inequality(self):
        """Test MarketData inequality comparison."""
        timestamp = datetime.now()
        
        data1 = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.0950"),
            ask=Decimal("1.0952"),
            last=Decimal("1.0951"),
            high=Decimal("1.0960"),
            low=Decimal("1.0940"),
            open=Decimal("1.0945"),
            close=Decimal("1.0951"),
            volume=Decimal("1000000")
        )
        
        data2 = MarketData(
            symbol="GBPUSD",  # Different symbol
            timestamp=timestamp,
            bid=Decimal("1.2500"),
            ask=Decimal("1.2502"),
            last=Decimal("1.2501"),
            high=Decimal("1.2510"),
            low=Decimal("1.2490"),
            open=Decimal("1.2495"),
            close=Decimal("1.2501"),
            volume=Decimal("500000")
        )
        
        assert data1 != data2
        
    def test_decimal_precision(self):
        """Test that Decimal types maintain precision."""
        timestamp = datetime.now()
        
        data = MarketData(
            symbol="EURUSD",
            timestamp=timestamp,
            bid=Decimal("1.095012345"),  # High precision
            ask=Decimal("1.095212345"),
            last=Decimal("1.095112345"),
            high=Decimal("1.096012345"),
            low=Decimal("1.094012345"),
            open=Decimal("1.094512345"),
            close=Decimal("1.095112345"),
            volume=Decimal("1000000.123")
        )
        
        # Verify precision is maintained
        assert str(data.bid) == "1.095012345"
        assert str(data.volume) == "1000000.123"

    def test_zero_bid_ask_calculations(self):
        """Ensure spread and mid_price are calculated when prices include zeros."""
        timestamp = datetime.now()

        data = MarketData(
            symbol="ZERO",
            timestamp=timestamp,
            bid=Decimal("0"),
            ask=Decimal("1"),
            last=Decimal("0.5"),
            high=Decimal("1"),
            low=Decimal("0"),
            open=Decimal("0"),
            close=Decimal("1"),
            volume=Decimal("100")
        )

        assert data.spread == Decimal("1")
        assert data.mid_price == Decimal("0.5")


if __name__ == "__main__":
    pytest.main([__file__])

