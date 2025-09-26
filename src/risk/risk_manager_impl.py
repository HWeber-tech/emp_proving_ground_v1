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
from src.config.risk.risk_config import RiskConfig
from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig
from src.core.coercion import coerce_float

logger = logging.getLogger(__name__)

__all__ = ["RiskManagerImpl", "PositionEntry"]


def _to_float(value: object | None, *, default: float = 0.0) -> float:
    """Coerce heterogeneous inputs to ``float`` at API boundaries."""

    return coerce_float(value, default=default)


class PositionEntry(TypedDict):
    symbol: NotRequired[str]  # stored in dict key; optional here
    size: float
    entry_price: float
    entry_time: datetime
    current_price: NotRequired[float]
    stop_loss_pct: NotRequired[float]


class PositionInput(TypedDict):
    symbol: str
    size: float | Decimal
    entry_price: float | Decimal
    stop_loss_pct: NotRequired[float | Decimal]


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

    def __init__(
        self,
        initial_balance: float | Decimal = 10000.0,
        risk_config: RiskConfig | None = None,
    ) -> None:
        """
        Initialize the risk manager with configuration.

        Args:
            initial_balance: Starting account balance
            risk_config: Optional canonical risk configuration overrides
        """
        initial_balance_float = _to_float(initial_balance)

        self._risk_config = risk_config or RiskConfig()

        self._min_position_size = float(self._risk_config.min_position_size)
        self._max_position_size = float(self._risk_config.max_position_size)
        self._mandatory_stop_loss = bool(self._risk_config.mandatory_stop_loss)
        self._research_mode = bool(self._risk_config.research_mode)

        self.config = RealRiskConfig(
            max_position_risk=float(self._risk_config.max_risk_per_trade_pct),
            max_total_exposure=float(self._risk_config.max_total_exposure_pct),
            max_drawdown=float(self._risk_config.max_drawdown_pct),
            max_leverage=float(self._risk_config.max_leverage),
            equity=initial_balance_float,
        )

        self.risk_manager = RealRiskManager(self.config)

        # Track current positions
        self.positions: Dict[str, PositionEntry] = {}
        self.account_balance: float = initial_balance_float
        self.peak_balance: float = self.account_balance
        # Risk per trade constant used across validation and sizing routines.
        self._risk_per_trade: float = float(self._risk_config.max_risk_per_trade_pct)
        # Drawdown multiplier throttles exposure during recoveries (1.0 == full risk).
        self._drawdown_multiplier: float = 1.0

        self._recompute_drawdown_multiplier()

        self.risk_manager.update_equity(self.account_balance)

        logger.info(f"RiskManagerImpl initialized with balance: ${self.account_balance:.2f}")

    def _recompute_drawdown_multiplier(self) -> None:
        """Adjust exposure multiplier based on drawdown severity."""
        if self.peak_balance <= 0:
            self._drawdown_multiplier = 1.0
            return

        drawdown = max(0.0, (self.peak_balance - self.account_balance) / self.peak_balance)

        if self.config.max_drawdown <= 0:
            # Avoid division by zero – treat as unlimited headroom.
            self._drawdown_multiplier = 1.0
            return

        normalized = max(0.0, min(1.0, drawdown / self.config.max_drawdown))
        # Preserve at least a quarter of the baseline risk budget so automation can recover.
        self._drawdown_multiplier = max(0.25, 1.0 - normalized)

    def _compute_risk_budget(self) -> float:
        """Return the dollar risk budget after drawdown throttling."""
        baseline_risk = min(self._risk_per_trade, self.config.max_position_risk)
        return self.account_balance * baseline_risk * self._drawdown_multiplier

    def _resolve_position_price(self, entry: PositionEntry) -> float:
        """Resolve the current price for a tracked position."""

        current = entry.get("current_price")
        if current is not None:
            resolved = _to_float(current)
            if resolved > 0:
                return resolved
        return max(0.0, _to_float(entry.get("entry_price", 0.0)))

    def _resolve_position_stop_loss(self, entry: PositionEntry) -> float:
        """Resolve the effective stop-loss percentage for a tracked position."""

        raw = entry.get("stop_loss_pct", self._risk_per_trade)
        resolved = _to_float(raw)
        return resolved if resolved > 0 else self._risk_per_trade

    def _compute_position_risk(self, entry: PositionEntry) -> float:
        """Compute risk exposure for a stored position."""

        price = self._resolve_position_price(entry)
        stop_loss = self._resolve_position_stop_loss(entry)
        size = entry.get("size", 0.0)
        return max(0.0, _to_float(size)) * price * stop_loss

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
            raw_stop_loss = position.get("stop_loss_pct")
            stop_loss_pct = _to_float(raw_stop_loss) if raw_stop_loss is not None else 0.0

            # Validate basic parameters
            if size <= 0:
                logger.warning(f"Invalid position size: {size}")
                return False

            if entry_price <= 0:
                logger.warning(f"Invalid entry price: {entry_price}")
                return False

            if size < self._min_position_size:
                logger.warning(
                    "Position below configured minimum size: %s < %s",
                    size,
                    self._min_position_size,
                )
                return False

            if size > self._max_position_size:
                logger.warning(
                    "Position exceeds configured maximum size: %s > %s",
                    size,
                    self._max_position_size,
                )
                return False

            if self._mandatory_stop_loss and not self._research_mode and stop_loss_pct <= 0:
                logger.warning("Stop loss missing while mandatory stop losses enabled")
                return False

            effective_stop_loss = stop_loss_pct if stop_loss_pct > 0 else self._risk_per_trade

            # Basic risk check: risk per trade capped by config
            risk_amount = size * entry_price * effective_stop_loss
            max_allowed_risk = self._compute_risk_budget()

            if max_allowed_risk <= 0:
                logger.warning("Risk budget unavailable; rejecting position for %s", symbol)
                return False

            is_valid = risk_amount <= max_allowed_risk

            # Evaluate aggregate exposure impact when within local budget.
            if is_valid:
                projected_risk: Dict[str, float] = {
                    sym: self._compute_position_risk(pos) for sym, pos in self.positions.items()
                }
                projected_risk[symbol] = projected_risk.get(symbol, 0.0) + risk_amount
                aggregate_risk = self.risk_manager.assess_risk(projected_risk)
                if aggregate_risk > 1.0:
                    snapshot = self.risk_manager.last_snapshot
                    logger.warning(
                        "Position rejected due to aggregate risk breach: %s (snapshot=%s)",
                        symbol,
                        snapshot,
                    )
                    return False

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
            stop_loss_pct = max(_to_float(signal.get("stop_loss_pct", 0.05)), 1e-9)

            # Calculate win rate based on confidence
            win_rate = max(0.1, min(0.9, confidence))

            # Use simple historical performance assumptions for Kelly calculation
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss

            # Kelly fraction: p - q/b where b = avg_win/avg_loss
            b = avg_win / max(avg_loss, 1e-9)
            kelly_fraction = max(0.0, min(1.0, win_rate - (1.0 - win_rate) / b))

            risk_budget = self._compute_risk_budget()
            position_size = risk_budget / stop_loss_pct

            # Apply Kelly fraction
            final_size = position_size * kelly_fraction

            # Enforce configured boundaries.
            bounded_size = max(self._min_position_size, min(self._max_position_size, final_size))

            logger.info(
                "Calculated position size: %s size=%.2f (bounded to %.2f)",
                symbol,
                final_size,
                bounded_size,
            )

            return max(self._min_position_size, bounded_size)

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
        self.peak_balance = max(self.peak_balance, self.account_balance)
        self._recompute_drawdown_multiplier()
        self.risk_manager.update_equity(self.account_balance)
        logger.info(
            "Account balance updated: $%.2f (peak: $%.2f, drawdown multiplier: %.2f)",
            self.account_balance,
            self.peak_balance,
            self._drawdown_multiplier,
        )

    def add_position(
        self,
        symbol: str,
        size: float | Decimal,
        entry_price: float | Decimal,
        stop_loss_pct: float | Decimal | None = None,
    ) -> None:
        """
        Add a new position to track.

        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
            stop_loss_pct: Optional configured stop loss for aggregate risk checks
        """
        entry: PositionEntry = {
            "size": _to_float(size),
            "entry_price": _to_float(entry_price),
            "entry_time": datetime.now(),
        }

        if stop_loss_pct is not None:
            entry["stop_loss_pct"] = max(0.0, _to_float(stop_loss_pct))
        else:
            entry["stop_loss_pct"] = self._risk_per_trade

        self.positions[symbol] = entry

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
        risk_map = {s: self._compute_position_risk(p) for s, p in self.positions.items()}
        assessed_risk = self.risk_manager.assess_risk(risk_map)

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
        projected_risk = {
            sym: self._compute_position_risk(pos) for sym, pos in self.positions.items()
        }
        total_risk_amount = sum(projected_risk.values())
        assessed_risk = self.risk_manager.assess_risk(projected_risk)

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
            "risk_amount": self._compute_position_risk(position),
        }

    # Protocol-compliant methods (src.core.interfaces.RiskManager)
    def evaluate_portfolio_risk(
        self,
        positions: Mapping[str, float],
        context: Mapping[str, object] | None = None,
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
        constraints: Mapping[str, object] | None = None,
    ) -> Mapping[str, float]:
        # Minimal adapter: preserve existing allocations (no-op)
        return dict(positions)

    def update_limits(self, limits: Mapping[str, object]) -> None:
        # Accept float|Decimal, coerce using _to_float
        if "max_position_risk" in limits or "max_risk_per_trade_pct" in limits:
            candidate = limits.get("max_risk_per_trade_pct", limits.get("max_position_risk", 0.0))
            resolved = max(0.0, _to_float(candidate))
            if resolved > 0:
                self._risk_per_trade = resolved
                self.config.max_position_risk = resolved

        if "max_drawdown" in limits:
            drawdown = max(0.0, _to_float(limits["max_drawdown"]))
            self.config.max_drawdown = drawdown
            if drawdown > 0:
                self.config.max_total_exposure = drawdown

        if "max_total_exposure_pct" in limits:
            total_exposure = max(0.0, _to_float(limits["max_total_exposure_pct"]))
            if total_exposure > 0:
                self.config.max_total_exposure = total_exposure
                self.config.max_drawdown = total_exposure

        if "max_leverage" in limits:
            leverage = max(0.0, _to_float(limits["max_leverage"]))
            if leverage > 0:
                self.config.max_leverage = leverage

        if "min_position_size" in limits:
            self._min_position_size = max(0.0, _to_float(limits["min_position_size"]))

        if "max_position_size" in limits:
            candidate_max = max(0.0, _to_float(limits["max_position_size"]))
            self._max_position_size = max(candidate_max, self._min_position_size)

        if "mandatory_stop_loss" in limits:
            self._mandatory_stop_loss = bool(limits["mandatory_stop_loss"])

        if "research_mode" in limits:
            self._research_mode = bool(limits["research_mode"])

        self._recompute_drawdown_multiplier()


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
            "stop_loss_pct": 0.02,
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
