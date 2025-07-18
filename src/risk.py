"""
Risk Management Module - v2.0 Implementation

This module implements the robust risk management system as specified in v2.0,
using Decimal for all financial calculations and comprehensive validation.
"""

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import TYPE_CHECKING, Any, Dict, Optional


from src.core import Instrument, InstrumentProvider, RiskConfig

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from pnl import EnhancedPosition
    from simulation import Order

# Configure decimal precision for financial calculations
getcontext().prec = 12
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of risk validation"""

    is_valid: bool
    reason: str
    risk_metadata: Optional[Dict[str, Any]] = None


class RiskManager:
    """
    v2.0 Risk Manager with proper Decimal handling and comprehensive validation.

    This class implements the risk management logic from the original unified file,
    ensuring all financial calculations use Decimal for precision and accuracy.
    """

    def __init__(self, config: RiskConfig, instrument_provider: InstrumentProvider):
        """
        Initialize the risk manager.

        Args:
            config: Risk configuration with proper Decimal values
            instrument_provider: Provider for instrument metadata
        """
        self.config = config
        self.instrument_provider = instrument_provider
        logger.info("RiskManager initialized with v2.0 specifications")

    def calculate_position_size(
        self,
        account_equity: Decimal,
        stop_loss_pips: Decimal,
        instrument: Instrument,
        account_currency: str = "USD",
    ) -> int:
        """
        Calculate position size based on risk management rules.

        Args:
            account_equity: Current account equity
            stop_loss_pips: Stop loss in pips
            instrument: Trading instrument
            account_currency: Account currency

        Returns:
            Position size in units
        """
        if account_equity <= 0:
            logger.warning("Account equity must be positive")
            return 0

        if stop_loss_pips <= 0:
            logger.warning("Stop loss must be positive")
            return 0

        # Calculate risk amount (2% of equity)
        risk_amount = account_equity * self.config.max_risk_per_trade_pct

        # Calculate pip value
        pip_value = self._calculate_pip_value(instrument, account_currency)
        if pip_value <= 0:
            logger.warning("Invalid pip value")
            return 0

        # Calculate position size based on risk
        position_size = int(risk_amount / (stop_loss_pips * pip_value))

        # Apply position size limits
        position_size = max(self.config.min_position_size, position_size)
        position_size = min(self.config.max_position_size, position_size)

        logger.debug(f"Calculated position size: {position_size} units")
        return position_size

    def _calculate_pip_value(
        self, instrument: Instrument, account_currency: str
    ) -> Decimal:
        """
        Calculate the value of one pip in account currency.

        Args:
            instrument: Trading instrument
            account_currency: Account currency

        Returns:
            Pip value in account currency
        """
        # For now, use a simplified calculation
        # In a full implementation, this would use the CurrencyConverter
        if account_currency == "USD":
            if "USD" in instrument.symbol:
                return Decimal("10.0")  # $10 per pip for USD pairs
            else:
                return Decimal("10.0")  # Simplified for now
        else:
            # Would need proper currency conversion
            return Decimal("10.0")

    def validate_order(
        self,
        proposed_order: "Order",
        account_state: Dict,
        open_positions: Dict[str, "EnhancedPosition"],
    ) -> ValidationResult:
        """
        Validate a proposed order against risk management rules.

        Args:
            proposed_order: Order to validate
            account_state: Current account state
            open_positions: Currently open positions

        Returns:
            Validation result with details
        """
        # Get instrument
        instrument = self.instrument_provider.get_instrument(proposed_order.symbol)
        if not instrument:
            return ValidationResult(
                is_valid=False, reason=f"Instrument {proposed_order.symbol} not found"
            )

        # Check leverage limits
        if not self._validate_leverage(proposed_order, account_state, open_positions):
            return ValidationResult(
                is_valid=False, reason="Order exceeds maximum leverage limits"
            )

        # Check exposure limits
        if not self._validate_exposure(proposed_order, account_state, open_positions):
            return ValidationResult(
                is_valid=False, reason="Order exceeds maximum exposure limits"
            )

        # Check drawdown limits
        if not self._validate_drawdown(account_state):
            return ValidationResult(
                is_valid=False, reason="Account drawdown exceeds maximum allowed"
            )

        # Check position size limits
        if not self._validate_position_size(proposed_order):
            return ValidationResult(
                is_valid=False, reason="Position size outside allowed limits"
            )

        return ValidationResult(
            is_valid=True,
            reason="Order approved",
            risk_metadata={
                "leverage_used": self._calculate_leverage_used(
                    proposed_order, account_state, open_positions
                ),
                "exposure_pct": self._calculate_exposure_pct(
                    proposed_order, account_state, open_positions
                ),
                "risk_per_trade_pct": self._calculate_risk_per_trade_pct(
                    proposed_order, account_state
                ),
            },
        )

    def _validate_leverage(
        self,
        order: "Order",
        account_state: Dict,
        open_positions: Dict[str, "EnhancedPosition"],
    ) -> bool:
        """Validate leverage limits."""
        current_leverage = self._calculate_leverage_used(
            order, account_state, open_positions
        )
        return current_leverage <= self.config.max_leverage

    def _validate_exposure(
        self,
        order: "Order",
        account_state: Dict,
        open_positions: Dict[str, "EnhancedPosition"],
    ) -> bool:
        """Validate exposure limits."""
        exposure_pct = self._calculate_exposure_pct(
            order, account_state, open_positions
        )
        return exposure_pct <= self.config.max_total_exposure_pct

    def _validate_drawdown(self, account_state: Dict) -> bool:
        """Validate drawdown limits."""
        if "drawdown_pct" not in account_state:
            return True  # No drawdown information available

        return account_state["drawdown_pct"] <= self.config.max_drawdown_pct

    def _validate_position_size(self, order: "Order") -> bool:
        """Validate position size limits."""
        return (
            self.config.min_position_size
            <= abs(order.quantity)
            <= self.config.max_position_size
        )

    def _calculate_leverage_used(
        self,
        order: "Order",
        account_state: Dict,
        open_positions: Dict[str, "EnhancedPosition"],
    ) -> Decimal:
        """Calculate total leverage used."""
        total_exposure = Decimal("0")
        account_equity = Decimal(str(account_state.get("equity", 0)))

        # Add exposure from new order
        instrument = self.instrument_provider.get_instrument(order.symbol)
        if instrument:
            total_exposure += (
                Decimal(str(abs(order.quantity))) * instrument.contract_size
            )

        # Add exposure from existing positions
        for position in open_positions.values():
            instrument = self.instrument_provider.get_instrument(position.symbol)
            if instrument:
                total_exposure += (
                    Decimal(str(abs(position.quantity))) * instrument.contract_size
                )

        if account_equity <= 0:
            return Decimal("0")

        return total_exposure / account_equity

    def validate_position(
        self, position: "EnhancedPosition", instrument: Instrument, equity: Decimal
    ) -> bool:
        """
        Validate a position against risk management rules.

        Args:
            position: Position to validate
            instrument: Trading instrument
            equity: Current account equity

        Returns:
            True if position is valid, False otherwise
        """
        try:
            # Check position size limits
            if not (
                self.config.min_position_size
                <= abs(position.quantity)
                <= self.config.max_position_size
            ):
                logger.warning(f"Position size {position.quantity} outside limits")
                return False

            # Check leverage limits
            position_value = position.get_position_value(
                Decimal(str(instrument.contract_size))
            )
            leverage_used = position_value / equity if equity > 0 else Decimal("0")

            if leverage_used > self.config.max_leverage:
                logger.warning(
                    f"Leverage {leverage_used} exceeds maximum {self.config.max_leverage}"
                )
                return False

            # Check exposure limits
            exposure_pct = position_value / equity if equity > 0 else Decimal("0")
            if exposure_pct > self.config.max_total_exposure_pct:
                logger.warning(
                    f"Exposure {exposure_pct} exceeds maximum {self.config.max_total_exposure_pct}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False

    def _calculate_exposure_pct(
        self,
        order: "Order",
        account_state: Dict,
        open_positions: Dict[str, "EnhancedPosition"],
    ) -> Decimal:
        """Calculate total exposure as percentage of equity."""
        total_exposure = Decimal("0")
        account_equity = Decimal(str(account_state.get("equity", 0)))

        # Add exposure from new order
        instrument = self.instrument_provider.get_instrument(order.symbol)
        if instrument:
            total_exposure += (
                Decimal(str(abs(order.quantity))) * instrument.contract_size
            )

        # Add exposure from existing positions
        for position in open_positions.values():
            instrument = self.instrument_provider.get_instrument(position.symbol)
            if instrument:
                total_exposure += (
                    Decimal(str(abs(position.quantity))) * instrument.contract_size
                )

        if account_equity <= 0:
            return Decimal("0")

        return total_exposure / account_equity

    def _calculate_risk_per_trade_pct(
        self, order: "Order", account_state: Dict
    ) -> Decimal:
        """Calculate risk per trade as percentage of equity."""
        account_equity = Decimal(str(account_state.get("equity", 0)))
        if account_equity <= 0:
            return Decimal("0")

        # Simplified risk calculation
        # In a full implementation, this would consider stop loss and pip value
        instrument = self.instrument_provider.get_instrument(order.symbol)
        if instrument:
            risk_amount = (
                Decimal(str(abs(order.quantity)))
                * instrument.contract_size
                * Decimal("0.01")
            )  # 1% risk
            return risk_amount / account_equity

        return Decimal("0")
