"""
Risk Manager Implementation
==========================

Complete implementation of the Risk Manager adapter for the EMP system.
Provides risk management capabilities with position sizing and validation,
aligned to canonical imports and types.
"""

import logging
from typing import Dict, TypedDict, NotRequired, Mapping
from decimal import Decimal
from datetime import datetime
import asyncio

from src.core.types import JSONObject
from src.core.interfaces import RiskManager as RiskManagerProtocol
from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig

logger = logging.getLogger(__name__)

__all__ = ["RiskManagerImpl", "PositionEntry"]

def _to_float(v: float | Decimal) -> float:
    """Coerce float|Decimal to float at API boundaries."""
    return float(v)

class PositionEntry(TypedDict):
    symbol: NotRequired[str]  # stored in dict key; optional here
    size: float
    entry_price: float
    entry_time: datetime
    current_price: NotRequired[float]

class PositionInput(TypedDict):
    symbol: str
    size: float | Decimal
    entry_price: float | Decimal

class SignalInput(TypedDict):
    symbol: str
    confidence: float
    stop_loss_pct: float | Decimal

class RiskManagerImpl(RiskManagerProtocol):
    """
    Adapter providing higher-level risk utilities on top of RealRiskManager.
    This class maintains minimal local state and delegates portfolio assessment
    to the canonical RealRiskManager where applicable.
    """

    def __init__(self, initial_balance: float | Decimal = 10000.0) -> None:
        """
        Initialize the risk manager with configuration.

        Args:
            initial_balance: Starting account balance
        """
        self.config = RealRiskConfig(
            max_position_risk=0.02,  # 2% max risk per position
            max_drawdown=0.25,       # 25% max drawdown
        )

        self.risk_manager = RealRiskManager(self.config)

        # Track current positions
        self.positions: Dict[str, PositionEntry] = {}
        self.account_balance: float = _to_float(initial_balance)

        logger.info(f"RiskManagerImpl initialized with balance: ${self.account_balance:.2f}")

    async def validate_position(self, position: PositionInput) -> bool:
        """
        Validate if position meets risk criteria.

        Args:
            position: Position details including symbol, size, entry_price

        Returns:
            True if position is valid, False otherwise
        """
        try:
            symbol = position.get("symbol", "")
            size = _to_float(position.get("size", 0.0))
            entry_price = _to_float(position.get("entry_price", 0.0))

            # Validate basic parameters
            if size <= 0:
                logger.warning(f"Invalid position size: {size}")
                return False

            if entry_price <= 0:
                logger.warning(f"Invalid entry price: {entry_price}")
                return False

            # Basic risk check: risk per trade capped by config
            risk_per_trade = 0.02  # 2% risk per trade
            risk_amount = size * risk_per_trade
            max_allowed_risk = self.account_balance * self.config.max_position_risk

            is_valid = risk_amount <= max_allowed_risk

            if is_valid:
                logger.info(f"Position validated: {symbol} size={size}")
            else:
                logger.warning(f"Position rejected: {symbol} size={size}")

            return is_valid

        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return False

    async def calculate_position_size(self, signal: SignalInput) -> float:
        """
        Calculate appropriate position size for signal using Kelly-like sizing.

        Args:
            signal: Trading signal with risk parameters

        Returns:
            Position size in base currency
        """
        try:
            # Extract signal parameters
            symbol = signal.get("symbol", "")
            confidence = float(signal.get("confidence", 0.5))
            stop_loss_pct = _to_float(signal.get("stop_loss_pct", 0.05))

            # Calculate win rate based on confidence
            win_rate = max(0.1, min(0.9, confidence))

            # Use simple historical performance assumptions for Kelly calculation
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss

            # Kelly fraction: p - q/b where b = avg_win/avg_loss
            b = avg_win / max(avg_loss, 1e-9)
            kelly_fraction = max(0.0, min(1.0, win_rate - (1.0 - win_rate) / b))

            # Calculate position size based on risk
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = (self.account_balance * risk_per_trade) / max(stop_loss_pct, 1e-9)

            # Apply Kelly fraction
            final_size = position_size * kelly_fraction

            logger.info(f"Calculated position size: {symbol} size={final_size:.2f}")

            return max(1000.0, final_size)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1000.0  # Default minimum position

    def update_account_balance(self, new_balance: float | Decimal) -> None:
        """
        Update the account balance.

        Args:
            new_balance: New account balance
        """
        self.account_balance = _to_float(new_balance)
        logger.info(f"Account balance updated: ${self.account_balance:.2f}")

    def add_position(self, symbol: str, size: float | Decimal, entry_price: float | Decimal) -> None:
        """
        Add a new position to track.

        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
        """
        self.positions[symbol] = {
            "size": _to_float(size),
            "entry_price": _to_float(entry_price),
            "entry_time": datetime.now(),
        }

        logger.info(f"Position added: {symbol} size={size} price={entry_price}")

    def update_position_value(self, symbol: str, current_price: float | Decimal) -> None:
        """
        Update position value with current price.

        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = _to_float(current_price)

    def get_risk_summary(self) -> JSONObject:
        """
        Get comprehensive risk summary.

        Returns:
            JSON object with current risk metrics
        """
        # Delegate to RealRiskManager for a portfolio-level assessment (if any)
        positions_sizes = {s: p["size"] for s, p in self.positions.items()}
        assessed_risk = self.risk_manager.assess_risk(positions_sizes)

        summary: JSONObject = {
            "account_balance": self.account_balance,
            "positions": float(len(self.positions)),
            "tracked_positions": list(self.positions.keys()),
            "assessed_risk": assessed_risk,
        }

        return summary

    def calculate_portfolio_risk(self) -> JSONObject:
        """
        Calculate current portfolio risk metrics.

        Returns:
            JSON object with portfolio risk metrics
        """
        total_size = sum(p["size"] for p in self.positions.values())
        # Simple aggregate: 2% risk per position size
        total_risk_amount = sum(p["size"] * 0.02 for p in self.positions.values())
        assessed_risk = self.risk_manager.assess_risk({s: p["size"] for s, p in self.positions.items()})

        return {
            "total_size": total_size,
            "risk_amount": total_risk_amount,
            "assessed_risk": assessed_risk,
        }

    def get_position_risk(self, symbol: str) -> JSONObject:
        """
        Get risk metrics for a specific position.

        Args:
            symbol: Trading symbol

        Returns:
            Position risk metrics as JSON
        """
        if symbol not in self.positions:
            return {}

        position = self.positions[symbol]
        current_price = position.get("current_price", position["entry_price"])

        return {
            "symbol": symbol,
            "size": position["size"],
            "entry_price": position["entry_price"],
            "current_price": current_price,
            "risk_amount": position["size"] * 0.02,  # 2% risk
        }

    # Protocol-compliant methods (src.core.interfaces.RiskManager)
    def evaluate_portfolio_risk(
        self,
        positions: Mapping[str, float],
        context: JSONObject | None = None,
    ) -> float:
        try:
            numeric_positions: Dict[str, float] = {k: float(v) for k, v in positions.items()}
            return self.risk_manager.assess_risk(numeric_positions)
        except Exception as e:
            logger.error(f"Error evaluating portfolio risk: {e}")
            return 0.0

    def propose_rebalance(
        self,
        positions: Mapping[str, float],
        constraints: JSONObject | None = None,
    ) -> Mapping[str, float]:
        # Minimal adapter: preserve existing allocations (no-op)
        return dict(positions)

    def update_limits(self, limits: Mapping[str, float | Decimal]) -> None:
        # Accept float|Decimal, coerce using _to_float
        if "max_position_risk" in limits:
            self.config.max_position_risk = _to_float(limits["max_position_risk"])
        if "max_drawdown" in limits:
            self.config.max_drawdown = _to_float(limits["max_drawdown"])


# Factory function for easy instantiation
def create_risk_manager(initial_balance: float | Decimal = 10000.0) -> RiskManagerImpl:
    """
    Create a new RiskManagerImpl instance.

    Args:
        initial_balance: Starting account balance

    Returns:
        Configured RiskManagerImpl instance
    """
    return RiskManagerImpl(initial_balance)

if __name__ == "__main__":
    async def main() -> None:
        # Test the implementation
        print("Testing RiskManagerImpl...")

        risk_manager = create_risk_manager(10000.0)

        # Test position validation
        position: PositionInput = {
            "symbol": "EURUSD",
            "size": 10000.0,
            "entry_price": 1.1000,
        }

        is_valid = await risk_manager.validate_position(position)
        print(f"Position validation: {is_valid}")

        # Test position sizing
        signal: SignalInput = {
            "symbol": "EURUSD",
            "confidence": 0.7,
            "stop_loss_pct": 0.02,
        }

        position_size = await risk_manager.calculate_position_size(signal)
        print(f"Calculated position size: {position_size}")

        # Test risk summary
        summary = risk_manager.get_risk_summary()
        print(f"Risk summary: {summary}")

        print("RiskManagerImpl test completed successfully!")

    asyncio.run(main())
