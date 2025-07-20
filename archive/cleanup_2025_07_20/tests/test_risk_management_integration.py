"""
Test Suite for Risk Management Integration

Tests the complete risk management pipeline including:
- PositionSizer calculations
- RiskGateway validation
- TradingManager integration
- Trade rejection handling
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime

from src.core.events import TradeIntent
from src.trading.risk.position_sizer import PositionSizer
from src.trading.risk.risk_gateway import RiskGateway
from src.trading.trading_manager import TradingManager, MockPortfolioMonitor


class MockStrategyRegistry:
    """Mock strategy registry for testing."""
    
    def get_strategy_status(self, strategy_id: str) -> str:
        return "active"


class MockExecutionEngine:
    """Mock execution engine for testing."""
    
    def __init__(self):
        self.processed_orders = []
    
    async def process_order(self, order):
        self.processed_orders.append(order)


class TestPositionSizer:
    """Test PositionSizer functionality."""
    
    def test_fixed_fractional_calculation(self):
        """Test basic position sizing calculation."""
        sizer = PositionSizer(default_risk_per_trade=0.01)
        
        # Test normal case
        size = sizer.calculate_size_fixed_fractional(
            equity=10000,
            stop_loss_pips=50,
            pip_value=0.0001
        )
        
        # Expected: risk_amount = 10000 * 0.01 = 100
        # risk_per_unit = 50 * 0.0001 = 0.005
        # position_size = 100 / 0.005 = 20000
        assert size == 20000
    
    def test_kelly_placeholder(self):
        """Test Kelly Criterion placeholder raises NotImplementedError."""
        sizer = PositionSizer()
        
        with pytest.raises(NotImplementedError):
            sizer.calculate_size_kelly(0.6, 2.0, 10000)
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        sizer = PositionSizer()
        
        with pytest.raises(ValueError):
            sizer.calculate_size_fixed_fractional(
                equity=-1000,
                stop_loss_pips=50,
                pip_value=0.0001
            )
    
    def test_risk_parameters(self):
        """Test risk parameter retrieval."""
        sizer = PositionSizer(default_risk_per_trade=0.02)
        params = sizer.get_risk_parameters()
        
        assert params["default_risk_per_trade"] == 0.02
        assert params["method"] == "fixed_fractional"


class TestRiskGateway:
    """Test RiskGateway functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy_registry = MockStrategyRegistry()
        self.position_sizer = PositionSizer()
        self.portfolio_monitor = MockPortfolioMonitor()
        
        self.risk_gateway = RiskGateway(
            strategy_registry=self.strategy_registry,
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            max_open_positions=3,
            max_daily_drawdown=0.05
        )
    
    @pytest.mark.asyncio
    async def test_valid_trade_passes(self):
        """Test that valid trades pass all checks."""
        intent = TradeIntent(
            event_id="test_001",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            price=Decimal("1.1000"),
            metadata={
                "strategy_id": "test_strategy",
                "stop_loss_pips": 50
            }
        )
        
        portfolio_state = {
            "equity": 10000,
            "open_positions_count": 0,
            "current_daily_drawdown": 0.0,
            "pip_value": 0.0001
        }
        
        validated_intent = await self.risk_gateway.validate_trade_intent(
            intent, portfolio_state
        )
        
        assert validated_intent is not None
        assert validated_intent.quantity > 0
        assert "calculated_size" in validated_intent.metadata
        assert "stop_loss_price" in validated_intent.metadata
    
    @pytest.mark.asyncio
    async def test_strategy_inactive_rejection(self):
        """Test rejection when strategy is inactive."""
        intent = TradeIntent(
            event_id="test_002",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            metadata={
                "strategy_id": "inactive_strategy",
                "stop_loss_pips": 50
            }
        )
        
        portfolio_state = {
            "equity": 10000,
            "open_positions_count": 0,
            "current_daily_drawdown": 0.0,
            "pip_value": 0.0001
        }
        
        # Mock strategy registry to return inactive
        class InactiveStrategyRegistry:
            def get_strategy_status(self, strategy_id: str) -> str:
                return "inactive"
        
        risk_gateway = RiskGateway(
            strategy_registry=InactiveStrategyRegistry(),
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            max_open_positions=3,
            max_daily_drawdown=0.05
        )
        
        validated_intent = await risk_gateway.validate_trade_intent(
            intent, portfolio_state
        )
        
        assert validated_intent is None
    
    @pytest.mark.asyncio
    async def test_max_positions_rejection(self):
        """Test rejection when max positions reached."""
        intent = TradeIntent(
            event_id="test_003",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            metadata={
                "strategy_id": "test_strategy",
                "stop_loss_pips": 50
            }
        )
        
        portfolio_state = {
            "equity": 10000,
            "open_positions_count": 5,  # Exceeds max_open_positions=3
            "current_daily_drawdown": 0.0,
            "pip_value": 0.0001
        }
        
        validated_intent = await self.risk_gateway.validate_trade_intent(
            intent, portfolio_state
        )
        
        assert validated_intent is None
    
    @pytest.mark.asyncio
    async def test_drawdown_limit_rejection(self):
        """Test rejection when daily drawdown exceeded."""
        intent = TradeIntent(
            event_id="test_004",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            metadata={
                "strategy_id": "test_strategy",
                "stop_loss_pips": 50
            }
        )
        
        portfolio_state = {
            "equity": 10000,
            "open_positions_count": 0,
            "current_daily_drawdown": 0.06,  # Exceeds max_daily_drawdown=0.05
            "pip_value": 0.0001
        }
        
        validated_intent = await self.risk_gateway.validate_trade_intent(
            intent, portfolio_state
        )
        
        assert validated_intent is None
    
    def test_risk_limits(self):
        """Test risk limits retrieval."""
        limits = self.risk_gateway.get_risk_limits()
        
        assert limits["max_open_positions"] == 3
        assert limits["max_daily_drawdown"] == 0.05
        assert "position_sizer_config" in limits


class TestTradingManager:
    """Test TradingManager integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy_registry = MockStrategyRegistry()
        self.execution_engine = MockExecutionEngine()
        
        self.trading_manager = TradingManager(
            event_bus=None,  # Mock for now
            strategy_registry=self.strategy_registry,
            execution_engine=self.execution_engine,
            initial_equity=10000,
            risk_per_trade=0.01,
            max_open_positions=3,
            max_daily_drawdown=0.05
        )
    
    @pytest.mark.asyncio
    async def test_valid_trade_execution(self):
        """Test valid trade execution through TradingManager."""
        intent = TradeIntent(
            event_id="test_005",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            price=Decimal("1.1000"),
            metadata={
                "strategy_id": "test_strategy",
                "stop_loss_pips": 50
            }
        )
        
        await self.trading_manager.on_trade_intent(intent)
        
        # Check that trade was processed
        assert len(self.execution_engine.processed_orders) == 1
        processed_order = self.execution_engine.processed_orders[0]
        assert processed_order.symbol == "EURUSD"
        assert processed_order.action == "BUY"
    
    @pytest.mark.asyncio
    async def test_rejected_trade_no_execution(self):
        """Test rejected trades don't reach execution."""
        # Set up portfolio to trigger rejection
        self.trading_manager.portfolio_monitor.open_positions_count = 5
        
        intent = TradeIntent(
            event_id="test_006",
            timestamp=datetime.now(),
            source="test_strategy",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal("1000"),
            metadata={
                "strategy_id": "test_strategy",
                "stop_loss_pips": 50
            }
        )
        
        await self.trading_manager.on_trade_intent(intent)
        
        # Check that trade was rejected and not processed
        assert len(self.execution_engine.processed_orders) == 0
    
    def test_risk_status(self):
        """Test risk status reporting."""
        status = self.trading_manager.get_risk_status()
        
        assert "risk_limits" in status
        assert "portfolio_state" in status
        assert status["portfolio_state"]["equity"] == 10000


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
