"""
Simple Test for Risk Management Components

Tests basic functionality without complex imports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decimal import Decimal
import asyncio

# Test PositionSizer directly
def test_position_sizer():
    """Test PositionSizer functionality."""
    from trading.risk.position_sizer import PositionSizer
    
    sizer = PositionSizer(default_risk_per_trade=0.01)
    
    # Test basic calculation
    size = sizer.calculate_size_fixed_fractional(
        equity=10000,
        stop_loss_pips=50,
        pip_value=0.0001
    )
    
    print(f"Position size calculated: {size}")
    assert size == 20000  # 10000 * 0.01 / (50 * 0.0001) = 20000
    
    # Test risk parameters
    params = sizer.get_risk_parameters()
    print(f"Risk parameters: {params}")
    assert params["default_risk_per_trade"] == 0.01
    
    print("✅ PositionSizer tests passed")


# Test RiskGateway directly
async def test_risk_gateway():
    """Test RiskGateway functionality."""
    from trading.risk.risk_gateway import RiskGateway
    from trading.risk.position_sizer import PositionSizer
    
    class MockStrategyRegistry:
        def get_strategy_status(self, strategy_id: str) -> str:
            return "active"
    
    class MockPortfolioMonitor:
        def get_state(self):
            return {
                "equity": 10000,
                "open_positions_count": 0,
                "current_daily_drawdown": 0.0,
                "pip_value": 0.0001
            }
    
    position_sizer = PositionSizer()
    risk_gateway = RiskGateway(
        strategy_registry=MockStrategyRegistry(),
        position_sizer=position_sizer,
        portfolio_monitor=MockPortfolioMonitor(),
        max_open_positions=3,
        max_daily_drawdown=0.05
    )
    
    # Test risk limits
    limits = risk_gateway.get_risk_limits()
    print(f"Risk limits: {limits}")
    assert limits["max_open_positions"] == 3
    
    print("✅ RiskGateway tests passed")


# Test TradingManager directly
async def test_trading_manager():
    """Test TradingManager functionality."""
    from trading.trading_manager import TradingManager, MockPortfolioMonitor
    from trading.risk.position_sizer import PositionSizer
    
    class MockStrategyRegistry:
        def get_strategy_status(self, strategy_id: str) -> str:
            return "active"
    
    class MockExecutionEngine:
        def __init__(self):
            self.processed_orders = []
        
        async def process_order(self, order):
            self.processed_orders.append(order)
    
    execution_engine = MockExecutionEngine()
    trading_manager = TradingManager(
        event_bus=None,
        strategy_registry=MockStrategyRegistry(),
        execution_engine=execution_engine,
        initial_equity=10000,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_daily_drawdown=0.05
    )
    
    # Test risk status
    status = trading_manager.get_risk_status()
    print(f"TradingManager status: {status}")
    assert status["portfolio_state"]["equity"] == 10000
    
    print("✅ TradingManager tests passed")


async def main():
    """Run all tests."""
    print("🧪 Testing Risk Management Components...")
    
    test_position_sizer()
    await test_risk_gateway()
    await test_trading_manager()
    
    print("\n🎉 All risk management tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
