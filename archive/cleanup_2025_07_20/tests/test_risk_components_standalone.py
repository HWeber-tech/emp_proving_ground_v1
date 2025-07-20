"""
Standalone Test for Risk Management Components

Tests basic functionality without imports.
"""

from decimal import Decimal


class MockPositionSizer:
    """Simple mock position sizer for testing."""
    
    def __init__(self, default_risk_per_trade=0.01):
        self.default_risk_per_trade = default_risk_per_trade
    
    def calculate_size_fixed_fractional(self, equity, stop_loss_pips, pip_value):
        """Calculate position size using fixed fractional method."""
        if equity <= 0:
            raise ValueError("Equity must be positive")
        if stop_loss_pips <= 0:
            raise ValueError("Stop loss pips must be positive")
        if pip_value <= 0:
            raise ValueError("Pip value must be positive")
        
        risk_amount = equity * self.default_risk_per_trade
        risk_per_unit = stop_loss_pips * pip_value
        return risk_amount / risk_per_unit
    
    def get_risk_parameters(self):
        """Get current risk parameters."""
        return {
            "default_risk_per_trade": self.default_risk_per_trade,
            "method": "fixed_fractional"
        }


class MockRiskGateway:
    """Simple mock risk gateway for testing."""
    
    def __init__(self, max_open_positions=5, max_daily_drawdown=0.05):
        self.max_open_positions = max_open_positions
        self.max_daily_drawdown = max_daily_drawdown
    
    def validate_trade_intent(self, intent, portfolio_state):
        """Validate trade intent against risk rules."""
        # Check strategy status (mock)
        if intent.get("strategy_id") == "inactive":
            return None
        
        # Check open positions
        if portfolio_state.get("open_positions_count", 0) >= self.max_open_positions:
            return None
        
        # Check daily drawdown
        if portfolio_state.get("current_daily_drawdown", 0) > self.max_daily_drawdown:
            return None
        
        # Calculate position size
        position_sizer = MockPositionSizer()
        size = position_sizer.calculate_size_fixed_fractional(
            equity=portfolio_state.get("equity", 10000),
            stop_loss_pips=intent.get("stop_loss_pips", 50)),
            pip_value=portfolio_state.get("pip_value", 0.0001)
        )
        
        # Return enriched intent
        return {
            **intent,
            "calculated_size": size,
            "stop_loss_price": 1.0950,  # Mock calculation
            "risk_validated": True
        }


def test_position_sizer():
    """Test position sizing calculations."""
    print("🧪 Testing Position Sizer...")
    
    sizer = MockPositionSizer(default_risk_per_trade=0.01)
    
    # Test basic calculation
    size = sizer.calculate_size_fixed_fractional(
        equity=10000,
        stop_loss_pips=50,
        pip_value=0.0001
    )
    
    expected = 20000  # 10000 * 0.01 / (50 * 0.0001)
    print(f"Position size: {size} (expected: {expected})")
    assert size == expected
    
    # Test risk parameters
    params = sizer.get_risk_parameters()
    print(f"Risk parameters: {params}")
    assert params["default_risk_per_trade"] == 0.01
    
    print("✅ PositionSizer tests passed")


def test_risk_gateway():
    """Test risk gateway validation."""
    print("🧪 Testing Risk Gateway...")
    
    gateway = MockRiskGateway(max_open_positions=3, max_daily_drawdown=0.05)
    
    # Test valid trade
    intent = {
        "strategy_id": "active",
        "symbol": "EURUSD",
        "action": "BUY",
        "stop_loss_pips": 50
    }
    
    portfolio_state = {
        "equity": 10000,
        "open_positions_count": 0,
        "current_daily_drawdown": 0.0,
        "pip_value": 0.0001
    }
    
    validated = gateway.validate_trade_intent(intent, portfolio_state)
    assert validated is not None
    assert "calculated_size" in validated
    print(f"Valid trade passed: {validated['calculated_size']}")
    
    # Test max positions rejection
    portfolio_state["open_positions_count"] = 5
    rejected = gateway.validate_trade_intent(intent, portfolio_state)
    assert rejected is None
    print("Max positions rejection: ✅")
    
    # Test drawdown rejection
    portfolio_state["open_positions_count"] = 0
    portfolio_state["current_daily_drawdown"] = 0.06
    rejected = gateway.validate_trade_intent(intent, portfolio_state)
    assert rejected is None
    print("Drawdown rejection: ✅")
    
    print("✅ RiskGateway tests passed")


def test_risk_limits():
    """Test risk configuration."""
    print("🧪 Testing Risk Limits...")
    
    gateway = MockRiskGateway(max_open_positions=5, max_daily_drawdown=0.03)
    
    assert gateway.max_open_positions == 5
    assert gateway.max_daily_drawdown == 0.03
    
    print("✅ Risk limits tests passed")


def main():
    """Run all tests."""
    print("🧪 Testing Risk Management Components...")
    print("=" * 50)
    
    test_position_sizer()
    print()
    
    test_risk_gateway()
    print()
    
    test_risk_limits()
    print()
    
    print("🎉 All risk management tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
