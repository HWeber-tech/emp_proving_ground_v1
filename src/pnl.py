"""
PnL Engine Module - v2.0 Implementation

This module implements the robust PnL calculation system as specified in v2.0,
using Decimal for all financial calculations and comprehensive trade tracking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any, Dict, List

from src.core import Instrument

# Configure decimal precision for financial calculations
getcontext().prec = 12
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """
    Immutable record of a trade transaction for audit trail.

    This class provides a complete audit trail of all trading activities,
    ensuring transparency and compliance with financial regulations.
    """

    timestamp: datetime
    trade_type: str  # 'OPEN', 'ADD', 'REDUCE', 'CLOSE', 'REVERSE', 'SWAP'
    quantity: int
    price: Decimal
    commission: Decimal
    slippage: Decimal
    swap_fee: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trade record data."""
        if self.quantity == 0 and self.trade_type != "SWAP":
            raise ValueError("Trade quantity cannot be zero for non-swap trades")
        if self.price < 0:
            raise ValueError("Trade price cannot be negative")
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        if self.slippage < 0:
            raise ValueError("Slippage cannot be negative")
        if self.swap_fee < 0:
            raise ValueError("Swap fee cannot be negative")


@dataclass
class EnhancedPosition:
    """
    Enhanced position with v2.0 features and complete audit trail.

    This class implements the position management logic from the original unified file,
    ensuring all financial calculations use Decimal for precision and accuracy.
    """

    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: Decimal
    entry_timestamp: datetime
    last_swap_time: datetime
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    max_adverse_excursion: Decimal = Decimal("0")
    max_favorable_excursion: Decimal = Decimal("0")
    trade_history: List[TradeRecord] = field(default_factory=list)

    def update(
        self,
        trade_quantity: int,
        trade_price: Decimal,
        commission: Decimal,
        slippage: Decimal,
        current_time: datetime,
        trade_type: str = "UNKNOWN",
    ) -> None:
        """
        Update position with new trade.

        Args:
            trade_quantity: Quantity of the trade (positive for buy, negative for sell)
            trade_price: Price of the trade
            commission: Commission paid
            slippage: Slippage incurred
            current_time: Timestamp of the trade
            trade_type: Type of trade ('OPEN', 'ADD', 'REDUCE', 'CLOSE', 'REVERSE')
        """
        # Create trade record
        trade_record = TradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            quantity=trade_quantity,
            price=trade_price,
            commission=commission,
            slippage=slippage,
        )
        self.trade_history.append(trade_record)

        # Calculate new position
        old_quantity = self.quantity
        old_avg_price = self.avg_price

        if trade_type in ["OPEN", "ADD"]:
            # Opening or adding to position
            if old_quantity == 0:
                # Opening new position
                self.quantity = trade_quantity
                self.avg_price = trade_price
                self.entry_timestamp = current_time
            else:
                # Adding to existing position
                total_quantity = old_quantity + trade_quantity
                self.avg_price = (
                    (old_quantity * old_avg_price) + (trade_quantity * trade_price)
                ) / total_quantity
                self.quantity = total_quantity

        elif trade_type in ["REDUCE", "CLOSE"]:
            # Reducing or closing position
            if abs(trade_quantity) > abs(old_quantity):
                raise ValueError(
                    f"Cannot close more than current position: {trade_quantity} vs {old_quantity}"
                )

            # Calculate realized PnL
            if trade_type == "CLOSE":
                # Full close
                pnl = (trade_price - old_avg_price) * old_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl
                self.quantity = 0
            else:
                # Partial close
                pnl = (trade_price - old_avg_price) * trade_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl
                self.quantity = old_quantity - trade_quantity

        elif trade_type == "REVERSE":
            # Reverse position (close old and open new)
            # First close existing position
            if old_quantity != 0:
                pnl = (trade_price - old_avg_price) * old_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl

            # Then open new position
            self.quantity = trade_quantity
            self.avg_price = trade_price
            self.entry_timestamp = current_time

        logger.debug(f"Position updated: {trade_type} {trade_quantity} @ {trade_price}")

    def update_unrealized_pnl(self, current_market_price: Decimal) -> None:
        """
        Update unrealized PnL and track MAE/MFE.

        Args:
            current_market_price: Current market price
        """
        if self.quantity == 0:
            self.unrealized_pnl = Decimal("0")
            return

        # Calculate unrealized PnL
        pnl = (current_market_price - self.avg_price) * self.quantity
        if self.quantity < 0:  # Short position
            pnl = -pnl

        self.unrealized_pnl = pnl

        # Update MAE/MFE
        if pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = pnl
        if pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = pnl

    def apply_swap_fee(self, current_time: datetime, instrument: Instrument) -> None:
        """
        Apply swap fee if past swap time.

        Args:
            current_time: Current timestamp
            instrument: Trading instrument
        """
        if self.quantity == 0:
            return

        # Parse swap time
        swap_time_s = str(getattr(instrument, "swap_time", "00:00"))
        swap_hour, swap_minute = map(int, swap_time_s.split(":"))
        swap_time = current_time.replace(
            hour=swap_hour, minute=swap_minute, second=0, microsecond=0
        )

        # Check if we're past swap time and it's a new day
        if current_time >= swap_time and current_time.date() > self.last_swap_time.date():
            # Apply appropriate swap rate
            if self.quantity > 0:  # Long position
                long_rate = float(getattr(instrument, "long_swap_rate", 0.0))
                swap_fee = Decimal(str(long_rate)) * abs(self.quantity)
            else:  # Short position
                short_rate = float(getattr(instrument, "short_swap_rate", 0.0))
                swap_fee = Decimal(str(short_rate)) * abs(self.quantity)

            # Add to trade history
            swap_record = TradeRecord(
                timestamp=current_time,
                trade_type="SWAP",
                quantity=0,
                price=Decimal("0"),
                commission=Decimal("0"),
                slippage=Decimal("0"),
                swap_fee=swap_fee,
            )
            self.trade_history.append(swap_record)

            # Update realized PnL
            self.realized_pnl -= swap_fee

            # Update last swap time
            self.last_swap_time = current_time

            logger.debug(f"Applied swap fee: {swap_fee}")

    def get_total_pnl(self) -> Decimal:
        """Get total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def get_total_cost(self) -> Decimal:
        """Get total transaction costs."""
        total_cost = Decimal("0")
        for trade in self.trade_history:
            total_cost += trade.commission + trade.slippage + trade.swap_fee
        return total_cost

    def get_trade_count(self) -> int:
        """Get total number of trades."""
        return len(self.trade_history)

    def get_position_value(self, current_price: Decimal) -> Decimal:
        """Get current position value."""
        return abs(self.quantity) * current_price

    def get_margin_required(self, instrument: Instrument, leverage: Decimal) -> Decimal:
        """Calculate margin required for this position."""
        position_value = self.get_position_value(self.avg_price)
        return position_value / leverage

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0

    def get_duration(self, current_time: datetime) -> float:
        """Get position duration in days."""
        if self.is_flat():
            return 0.0
        duration = current_time - self.entry_timestamp
        return duration.total_seconds() / (24 * 3600)  # Convert to days

    def get_summary(self) -> Dict[str, Any]:
        """Get position summary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": float(self.avg_price),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.get_total_pnl()),
            "max_adverse_excursion": float(self.max_adverse_excursion),
            "max_favorable_excursion": float(self.max_favorable_excursion),
            "trade_count": self.get_trade_count(),
            "total_cost": float(self.get_total_cost()),
            "is_long": self.is_long(),
            "is_short": self.is_short(),
            "is_flat": self.is_flat(),
        }
