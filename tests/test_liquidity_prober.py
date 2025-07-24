"""
Unit Tests for LiquidityProber
Tests PROBE-40.1: The LiquidityProber Engine functionality
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.trading.execution.liquidity_prober import LiquidityProber
from src.trading.integration.mock_ctrader_interface import CTraderInterface, OrderType, OrderSide


class MockOrder:
    """Mock order object for testing"""
    def __init__(self, order_id, status="filled", volume=0.001):
        self.order_id = order_id
        self.status = status
        self.volume = volume


class TestLiquidityProber:
    """Test suite for LiquidityProber functionality"""
    
    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker interface"""
        broker = Mock(spec=CTraderInterface)
        broker.place_order = AsyncMock()
        broker.cancel_order = AsyncMock()
        broker.get_orders = Mock()
        return broker
    
    @pytest.fixture
    def liquidity_prober(self, mock_broker):
        """Create LiquidityProber instance with mock broker"""
        config = {
            'probe_size': 0.001,
            'timeout_seconds': 1.0,
            'max_concurrent_probes': 3
        }
        return LiquidityProber(mock_broker, config)
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_success(self, liquidity_prober, mock_broker):
        """Test successful liquidity probing"""
        # Setup mock responses
        mock_broker.place_order.side_effect = [
            "order_1", "order_2", "order_3"
        ]
        
        # Mock order status progression
        orders = [
            MockOrder("order_1", "filled", 0.001),
            MockOrder("order_2", "filled", 0.0005),
            MockOrder("order_3", "filled", 0.001)
        ]
        mock_broker.get_orders.return_value = orders
        
        # Test probing
        price_levels = [1.1000, 1.1001, 1.1002]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        # Assertions
        assert len(results) == 3
        assert results[1.1000] == 0.001
        assert results[1.1001] == 0.0005
        assert results[1.1002] == 0.001
        
        # Verify broker calls
        assert mock_broker.place_order.call_count == 3
        assert mock_broker.cancel_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_timeout(self, liquidity_prober, mock_broker):
        """Test liquidity probing with timeout"""
        # Setup mock to simulate timeout
        async def slow_place_order(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return "order_1"
        
        mock_broker.place_order.side_effect = slow_place_order
        
        # Test with short timeout
        price_levels = [1.1000]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        # Should return empty results due to timeout
        assert results == {1.1000: 0.0}
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_partial_fill(self, liquidity_prober, mock_broker):
        """Test probing with partial fills"""
        # Setup mock responses
        mock_broker.place_order.return_value = "order_1"
        
        # Mock order with partial fill
        orders = [MockOrder("order_1", "partially_filled", 0.0003)]
        mock_broker.get_orders.return_value = orders
        
        price_levels = [1.1000]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        assert results[1.1000] == 0.0003
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_no_fill(self, liquidity_prober, mock_broker):
        """Test probing with no fills"""
        # Setup mock responses
        mock_broker.place_order.return_value = "order_1"
        
        # Mock order with no fill
        orders = [MockOrder("order_1", "new", 0.0)]
        mock_broker.get_orders.return_value = orders
        
        price_levels = [1.1000]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        assert results[1.1000] == 0.0
    
    def test_calculate_liquidity_confidence_score(self, liquidity_prober):
        """Test liquidity confidence score calculation"""
        # Test case 1: Perfect liquidity
        probe_results = {1.1000: 1.0, 1.1001: 1.0, 1.1002: 1.0}
        intended_volume = 1.0
        score = liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, intended_volume
        )
        assert 0.9 <= score <= 1.0
        
        # Test case 2: Insufficient liquidity
        probe_results = {1.1000: 0.1, 1.1001: 0.1, 1.1002: 0.1}
        intended_volume = 1.0
        score = liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, intended_volume
        )
        assert 0.0 <= score <= 0.5
        
        # Test case 3: Uneven distribution
        probe_results = {1.1000: 0.9, 1.1001: 0.05, 1.1002: 0.05}
        intended_volume = 1.0
        score = liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, intended_volume
        )
        assert 0.5 <= score <= 0.9
    
    def test_calculate_liquidity_confidence_score_edge_cases(self, liquidity_prober):
        """Test edge cases for confidence score calculation"""
        # Empty results
        score = liquidity_prober.calculate_liquidity_confidence_score({}, 1.0)
        assert score == 0.0
        
        # Zero intended volume
        score = liquidity_prober.calculate_liquidity_confidence_score({1.1000: 1.0}, 0.0)
        assert score == 0.0
        
        # Single level
        score = liquidity_prober.calculate_liquidity_confidence_score({1.1000: 1.0}, 1.0)
        assert score == 0.5
    
    def test_get_probe_summary(self, liquidity_prober):
        """Test probe summary generation"""
        probe_results = {1.1000: 0.5, 1.1001: 1.0, 1.1002: 0.0}
        summary = liquidity_prober.get_probe_summary(probe_results)
        
        assert summary["total_levels"] == 3
        assert summary["total_liquidity"] == 1.5
        assert summary["avg_liquidity"] == 0.5
        assert len(summary["best_levels"]) == 3
        assert summary["empty_levels"] == 1
    
    def test_get_probe_summary_empty(self, liquidity_prober):
        """Test probe summary with empty results"""
        summary = liquidity_prober.get_probe_summary({})
        
        assert summary["total_levels"] == 0
        assert summary["total_liquidity"] == 0.0
        assert summary["avg_liquidity"] == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, liquidity_prober):
        """Test health check functionality"""
        health = await liquidity_prober.health_check()
        
        assert health["status"] == "healthy"
        assert "config" in health
        assert health["config"]["probe_size"] == 0.001
        assert health["config"]["timeout_seconds"] == 1.0
        assert health["config"]["max_concurrent_probes"] == 3
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_sell_side(self, liquidity_prober, mock_broker):
        """Test liquidity probing for sell side"""
        mock_broker.place_order.return_value = "order_1"
        orders = [MockOrder("order_1", "filled", 0.001)]
        mock_broker.get_orders.return_value = orders
        
        price_levels = [1.1000]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "sell")
        
        # Verify sell side was used
        mock_broker.place_order.assert_called_with(
            symbol_name="EURUSD",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            volume=0.001,
            price=None
        )
        assert results[1.1000] == 0.001
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_concurrent_limits(self, liquidity_prober, mock_broker):
        """Test concurrent probe limits"""
        # Setup to test semaphore limits
        mock_broker.place_order.side_effect = [
            f"order_{i}" for i in range(5)
        ]
        
        orders = [MockOrder(f"order_{i}", "filled", 0.001) for i in range(5)]
        mock_broker.get_orders.return_value = orders
        
        price_levels = [1.1000, 1.1001, 1.1002, 1.1003, 1.1004]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        # Should complete all probes
        assert len(results) == 5
        assert mock_broker.place_order.call_count == 5
    
    @pytest.mark.asyncio
    async def test_probe_liquidity_broker_failure(self, liquidity_prober, mock_broker):
        """Test handling of broker failures"""
        # Simulate broker failure
        mock_broker.place_order.return_value = None
        
        price_levels = [1.1000]
        results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
        
        # Should handle gracefully
        assert results[1.1000] == 0.0


class TestLiquidityProberIntegration:
    """Integration tests for LiquidityProber with RiskGateway"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_liquidity_validation(self):
        """Test complete liquidity validation flow"""
        # This would test integration with RiskGateway
        # For now, we'll test the confidence score calculation
        from src.trading.risk.risk_gateway import RiskGateway
        from src.trading.risk.position_sizer import PositionSizer
        
        # Mock components
        mock_strategy_registry = Mock()
        mock_position_sizer = Mock(spec=PositionSizer)
        mock_portfolio_monitor = Mock()
        mock_broker = Mock(spec=CTraderInterface)
        
        # Setup mocks
        mock_position_sizer.calculate_size_fixed_fractional.return_value = 2.0
        mock_broker.place_order.return_value = "test_order"
        orders = [MockOrder("test_order", "filled", 0.001)]
        mock_broker.get_orders.return_value = orders
        
        # Create components
        liquidity_prober = LiquidityProber(mock_broker)
        risk_gateway = RiskGateway(
            mock_strategy_registry,
            mock_position_sizer,
            mock_portfolio_monitor,
            liquidity_prober=liquidity_prober,
            liquidity_probe_threshold=1.0
        )
        
        # Test large trade triggering liquidity probe
        from src.core.events import TradeIntent
        intent = TradeIntent(
            symbol="EURUSD",
            side="BUY",
            quantity=Decimal("2.0"),
            order_type="2",
            price=Decimal("1.1000")
        )
        
        portfolio_state = {
            "equity": 10000.0,
            "current_price": 1.1000,
            "pip_value": 0.0001
        }
        
        # This would test the full integration
        # For now, we'll test the confidence calculation
        probe_results = {1.1000: 1.0, 1.1001: 0.5, 1.1002: 0.3}
        confidence = liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, 2.0
        )
        
        assert confidence > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
