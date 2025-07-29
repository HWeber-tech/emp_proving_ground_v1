"""
Position Sizer v1.0 - Fixed-Fractional Risk Management

Implements TRADING-03: Dedicated utility for calculating optimal trade sizes
based on account equity and defined risk tolerance.
"""

import logging
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculates optimal trade sizes using fixed-fractional risk management.
    
    This utility decouples the "how much" decision from the "buy/sell" decision,
    ensuring consistent risk management across all trades.
    """
    
    def __init__(self, default_risk_per_trade: float = 0.01):
        """
        Initialize the PositionSizer with configurable risk parameters.
        
        Args:
            default_risk_per_trade: Percentage of equity to risk per trade (default: 1%)
        """
        self.default_risk_per_trade = Decimal(str(default_risk_per_trade))
        
        # Validation
        if not 0 < self.default_risk_per_trade <= Decimal('0.1'):
            raise ValueError("Risk per trade must be between 0 and 10%")
            
        logger.info(f"PositionSizer initialized with {self.default_risk_per_trade*100}% risk per trade")
    
    def calculate_size_fixed_fractional(
        self,
        equity: float,
        stop_loss_pips: int,
        pip_value: float,
        risk_per_trade: Optional[float] = None
    ) -> int:
        """
        Calculate position size using fixed-fractional risk management.
        
        Formula:
        risk_amount = equity * risk_per_trade
        risk_per_unit = stop_loss_pips * pip_value
        position_size = risk_amount / risk_per_unit
        
        Args:
            equity: Current account equity in base currency
            stop_loss_pips: Distance to stop loss in pips
            pip_value: Value of one pip in base currency
            risk_per_trade: Override default risk percentage (optional)
            
        Returns:
            Calculated position size as integer number of units
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Convert inputs to Decimal for precision
            equity_decimal = Decimal(str(equity))
            stop_loss_pips_decimal = Decimal(str(stop_loss_pips))
            pip_value_decimal = Decimal(str(pip_value))
            
            # Use provided risk or default
            risk_percentage = Decimal(str(risk_per_trade)) if risk_per_trade else self.default_risk_per_trade
            
            # Validate inputs
            if equity_decimal <= 0:
                raise ValueError("Equity must be positive")
            if stop_loss_pips_decimal <= 0:
                raise ValueError("Stop loss pips must be positive")
            if pip_value_decimal <= 0:
                raise ValueError("Pip value must be positive")
            if not 0 < risk_percentage <= Decimal('0.1'):
                raise ValueError("Risk percentage must be between 0 and 10%")
            
            # Calculate risk amount
            risk_amount = equity_decimal * risk_percentage
            
            # Calculate risk per unit
            risk_per_unit = stop_loss_pips_decimal * pip_value_decimal
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Round to nearest integer
            final_size = int(position_size.quantize(Decimal('1')))
            
            # Ensure minimum size of 1 unit
            final_size = max(1, final_size)
            
            logger.debug(
                f"Calculated position size: {final_size} units "
                f"(equity={equity}, stop_loss={stop_loss_pips}pips, "
                f"pip_value={pip_value}, risk={risk_percentage*100}%)"
            )
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise ValueError(f"Invalid position sizing parameters: {e}")
    
    def calculate_size_kelly(
        self,
        win_probability: float,
        win_loss_ratio: float,
        equity: float
    ) -> int:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly Formula: f = (bp - q) / b
        Where:
        - f = fraction of capital to wager
        - b = odds received on the wager (win_loss_ratio)
        - p = probability of winning (win_probability)
        - q = probability of losing (1 - win_probability)
        
        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss ratio
            equity: Current account equity
            
        Returns:
            Calculated position size as integer number of units
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Convert inputs to Decimal for precision
            p = Decimal(str(win_probability))
            b = Decimal(str(win_loss_ratio))
            equity_decimal = Decimal(str(equity))
            
            # Validate inputs
            if not 0 < p < 1:
                raise ValueError("Win probability must be between 0 and 1")
            if b <= 0:
                raise ValueError("Win/loss ratio must be positive")
            if equity_decimal <= 0:
                raise ValueError("Equity must be positive")
            
            # Calculate Kelly fraction
            q = Decimal('1') - p  # probability of losing
            kelly_fraction = (b * p - q) / b
            
            # Cap Kelly fraction at 25% for safety (Kelly can be aggressive)
            max_kelly = Decimal('0.25')
            kelly_fraction = min(kelly_fraction, max_kelly)
            
            # Ensure positive fraction (don't bet if Kelly is negative)
            kelly_fraction = max(kelly_fraction, Decimal('0'))
            
            # Calculate position size (simplified: assume 1 unit = 1% of equity)
            # In practice, this would need pip values and stop loss distances
            position_size = equity_decimal * kelly_fraction / Decimal('100')
            
            # Round to nearest integer
            final_size = int(position_size.quantize(Decimal('1')))
            
            # Ensure minimum size of 1 unit if Kelly suggests betting
            final_size = max(1, final_size) if kelly_fraction > 0 else 0
            
            logger.debug(
                f"Kelly position size: {final_size} units "
                f"(win_prob={win_probability}, win_loss_ratio={win_loss_ratio}, "
                f"kelly_fraction={kelly_fraction})"
            )
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}")
            raise ValueError(f"Invalid Kelly parameters: {e}")
    
    def get_risk_parameters(self) -> dict:
        """
        Get current risk parameters for monitoring/audit.
        
        Returns:
            Dictionary with current risk configuration
        """
        return {
            "default_risk_per_trade": float(self.default_risk_per_trade),
            "methods_available": ["fixed_fractional", "kelly_criterion"],
            "kelly_implementation": "implemented",
            "kelly_max_fraction": 0.25
        }
