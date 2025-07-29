"""
Unit tests for PositionSizer risk management module.
Tests both fixed-fractional and Kelly Criterion position sizing.
"""

import pytest
from decimal import Decimal
from src.trading.risk.position_sizer import PositionSizer


class TestPositionSizer:
    """Test suite for PositionSizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = PositionSizer(default_risk_per_trade=0.02)  # 2% risk
        
    def test_initialization(self):
        """Test PositionSizer initialization."""
        assert self.sizer.default_risk_per_trade == Decimal('0.02')
        
        # Test invalid risk percentage
        with pytest.raises(ValueError):
            PositionSizer(default_risk_per_trade=0.15)  # 15% too high
            
    def test_fixed_fractional_basic(self):
        """Test basic fixed-fractional position sizing."""
        equity = 10000.0
        stop_loss_pips = 20
        pip_value = 1.0
        
        size = self.sizer.calculate_size_fixed_fractional(
            equity=equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )
        
        # Expected: (10000 * 0.02) / (20 * 1.0) = 200 / 20 = 10 units
        assert size == 10
        
    def test_fixed_fractional_custom_risk(self):
        """Test fixed-fractional with custom risk percentage."""
        equity = 10000.0
        stop_loss_pips = 50
        pip_value = 0.5
        custom_risk = 0.01  # 1% risk
        
        size = self.sizer.calculate_size_fixed_fractional(
            equity=equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            risk_per_trade=custom_risk
        )
        
        # Expected: (10000 * 0.01) / (50 * 0.5) = 100 / 25 = 4 units
        assert size == 4
        
    def test_fixed_fractional_validation(self):
        """Test input validation for fixed-fractional sizing."""
        with pytest.raises(ValueError):
            self.sizer.calculate_size_fixed_fractional(
                equity=-1000,  # Negative equity
                stop_loss_pips=20,
                pip_value=1.0
            )
            
        with pytest.raises(ValueError):
            self.sizer.calculate_size_fixed_fractional(
                equity=10000,
                stop_loss_pips=0,  # Zero stop loss
                pip_value=1.0
            )
            
    def test_kelly_criterion_basic(self):
        """Test basic Kelly Criterion position sizing."""
        win_probability = 0.6  # 60% win rate
        win_loss_ratio = 1.5   # Average win 1.5x average loss
        equity = 10000.0
        
        size = self.sizer.calculate_size_kelly(
            win_probability=win_probability,
            win_loss_ratio=win_loss_ratio,
            equity=equity
        )
        
        # Kelly fraction = (1.5 * 0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333
        # Capped at 25% = 0.25
        # Position size = 10000 * 0.25 / 100 = 25 units
        assert size == 25
        
    def test_kelly_criterion_negative(self):
        """Test Kelly Criterion with negative expectation."""
        win_probability = 0.3  # 30% win rate
        win_loss_ratio = 1.0   # Equal win/loss amounts
        equity = 10000.0
        
        size = self.sizer.calculate_size_kelly(
            win_probability=win_probability,
            win_loss_ratio=win_loss_ratio,
            equity=equity
        )
        
        # Kelly fraction = (1.0 * 0.3 - 0.7) / 1.0 = -0.4 (negative)
        # Should return 0 (don't bet with negative expectation)
        assert size == 0
        
    def test_kelly_criterion_validation(self):
        """Test input validation for Kelly Criterion."""
        with pytest.raises(ValueError):
            self.sizer.calculate_size_kelly(
                win_probability=1.5,  # Invalid probability
                win_loss_ratio=1.0,
                equity=10000
            )
            
        with pytest.raises(ValueError):
            self.sizer.calculate_size_kelly(
                win_probability=0.6,
                win_loss_ratio=-1.0,  # Negative ratio
                equity=10000
            )
            
    def test_minimum_position_size(self):
        """Test minimum position size enforcement."""
        equity = 100.0  # Small equity
        stop_loss_pips = 100  # Large stop loss
        pip_value = 1.0
        
        size = self.sizer.calculate_size_fixed_fractional(
            equity=equity,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )
        
        # Calculated size would be very small, but minimum is 1
        assert size >= 1
        
    def test_risk_parameters(self):
        """Test risk parameters reporting."""
        params = self.sizer.get_risk_parameters()
        
        assert params["default_risk_per_trade"] == 0.02
        assert "fixed_fractional" in params["methods_available"]
        assert "kelly_criterion" in params["methods_available"]
        assert params["kelly_implementation"] == "implemented"
        assert params["kelly_max_fraction"] == 0.25


if __name__ == "__main__":
    pytest.main([__file__])

