"""
Risk Management Core for the EMP Proving Ground system.

This module provides:
- RiskManager: Position sizing and order validation
- ValidationResult: Risk validation results
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Any

from .core import RiskConfig, Instrument, InstrumentProvider, CurrencyConverter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of risk validation"""
    is_valid: bool
    reason: str
    risk_metadata: Optional[Dict[str, Any]] = None


class RiskManager:
    """Manages risk calculations and order validation"""
    
    def __init__(self, config: RiskConfig, instrument_provider: InstrumentProvider):
        self.config = config
        self.instrument_provider = instrument_provider
        self.currency_converter = CurrencyConverter()
    
    def calculate_position_size(self, account_equity: Decimal, stop_loss_pips: Decimal, 
                               instrument: Instrument, account_currency: str = "USD") -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            account_equity: Current account equity
            stop_loss_pips: Stop loss in pips
            instrument: Trading instrument
            account_currency: Account currency
            
        Returns:
            Position size in units
        """
        if stop_loss_pips <= 0:
            raise ValueError("Stop loss must be positive")
        
        # Calculate maximum risk amount
        max_risk_amount = account_equity * self.config.max_risk_per_trade_pct
        
        # Calculate pip value in account currency
        pip_value = self.currency_converter.calculate_pip_value(instrument, account_currency)
        
        # Calculate position size
        position_size = int(max_risk_amount / (stop_loss_pips * pip_value))
        
        # Apply position size limits
        position_size = max(self.config.min_position_size, position_size)
        position_size = min(self.config.max_position_size, position_size)
        
        return position_size
    
    def validate_order(self, proposed_order, account_state: Dict, 
                      open_positions: Dict[str, Any]) -> ValidationResult:
        """
        Validate a proposed order against risk parameters
        
        Args:
            proposed_order: The order to validate
            account_state: Current account state
            open_positions: Current open positions
            
        Returns:
            ValidationResult with validation outcome
        """
        try:
            # Get instrument
            instrument = self.instrument_provider.get_instrument(proposed_order.symbol)
            if not instrument:
                return ValidationResult(False, f"Instrument {proposed_order.symbol} not found")
            
            # Check position size limits
            if proposed_order.quantity < self.config.min_position_size:
                return ValidationResult(
                    False, 
                    f"Position size {proposed_order.quantity} below minimum {self.config.min_position_size}"
                )
            
            if proposed_order.quantity > self.config.max_position_size:
                return ValidationResult(
                    False,
                    f"Position size {proposed_order.quantity} above maximum {self.config.max_position_size}"
                )
            
            # Calculate total exposure
            total_exposure = self._calculate_total_exposure(open_positions, proposed_order)
            max_exposure = account_state.get('equity', 0) * self.config.max_total_exposure_pct
            
            if total_exposure > max_exposure:
                return ValidationResult(
                    False,
                    f"Total exposure {total_exposure} exceeds maximum {max_exposure}"
                )
            
            # Check leverage limits
            leverage = self._calculate_leverage(account_state, total_exposure)
            if leverage > self.config.max_leverage:
                return ValidationResult(
                    False,
                    f"Leverage {leverage} exceeds maximum {self.config.max_leverage}"
                )
            
            # Check stop loss requirement
            if self.config.mandatory_stop_loss and not proposed_order.stop_loss:
                return ValidationResult(
                    False,
                    "Stop loss is mandatory but not provided"
                )
            
            # All checks passed
            risk_metadata = {
                'total_exposure': float(total_exposure),
                'leverage': float(leverage),
                'position_size': proposed_order.quantity
            }
            
            return ValidationResult(True, "Order validated successfully", risk_metadata)
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return ValidationResult(False, f"Validation error: {str(e)}")
    
    def _calculate_total_exposure(self, open_positions: Dict[str, Any], proposed_order) -> Decimal:
        """Calculate total exposure including proposed order"""
        total_exposure = Decimal('0')
        
        # Add existing positions
        for position in open_positions.values():
            if hasattr(position, 'quantity') and hasattr(position, 'avg_price'):
                exposure = abs(position.quantity * position.avg_price)
                total_exposure += exposure
        
        # Add proposed order
        if proposed_order.price:
            proposed_exposure = abs(proposed_order.quantity * proposed_order.price)
            total_exposure += proposed_exposure
        
        return total_exposure
    
    def _calculate_leverage(self, account_state: Dict, total_exposure: Decimal) -> Decimal:
        """Calculate current leverage"""
        equity = Decimal(str(account_state.get('equity', 0)))
        if equity <= 0:
            return Decimal('0')
        
        return total_exposure / equity 