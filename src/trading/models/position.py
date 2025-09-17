"""
Position Models
===============

Data models for trading positions in the EMP Proving Ground system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(init=False)
class Position:
    """Represents a trading position with minimal, dependency-free semantics.

    Backward-compatible with existing fields, while providing a minimal API:
        - position_id: str | int
        - symbol: str
        - size: float            (alias of quantity)
        - entry_price: float     (alias of average_price)
        - current_price: float   (alias of market_price, defaults to entry_price)
        - value: float           (size * current_price)
        - unrealized_pnl: float  ((current_price - entry_price) * size)
        - update_price(new_price: float): None
    """

    # Minimal API field (optional)
    position_id: str | int | None = None

    # Existing core fields (kept for compatibility)
    symbol: str = ""
    quantity: float = 0.0
    average_price: float = 0.0

    # Existing accounting fields
    realized_pnl: float = 0.0

    # Pricing and timestamps
    market_price: Optional[float] = None
    last_updated: Optional[datetime] = None

    # Keep the original field name for compatibility; reads go through property or direct access
    unrealized_pnl: float = 0.0

    # Optional extended attributes for broader compatibility with portfolio code
    status: object | str | None = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None

    def __init__(
        self,
        symbol: str,
        size: Optional[float] = None,
        entry_price: Optional[float] = None,
        *,
        position_id: str | int | None = None,
        current_price: Optional[float] = None,
        quantity: Optional[float] = None,
        average_price: Optional[float] = None,
        market_price: Optional[float] = None,
        realized_pnl: float = 0.0,
        unrealized_pnl: Optional[float] = None,
        last_updated: Optional[datetime] = None,
    ) -> None:
        # Resolve canonical fields
        resolved_qty = quantity if quantity is not None else (size if size is not None else 0.0)
        resolved_avg = (
            average_price
            if average_price is not None
            else (entry_price if entry_price is not None else 0.0)
        )

        # Prefer explicit current/market over default-to-entry
        resolved_mkt = (
            market_price
            if market_price is not None
            else (current_price if current_price is not None else resolved_avg)
        )

        # Assign
        self.position_id = position_id
        self.symbol = symbol
        self.quantity = float(resolved_qty)
        self.average_price = float(resolved_avg)
        self.realized_pnl = float(realized_pnl)
        self.market_price = float(resolved_mkt) if resolved_mkt is not None else None

        # Initialize derived and timestamp
        if unrealized_pnl is not None:
            self.unrealized_pnl = float(unrealized_pnl)
        else:
            self.unrealized_pnl = (
                (self.current_price - self.entry_price) * self.size if self.quantity != 0 else 0.0
            )
        self.last_updated = last_updated or datetime.now()

    # Aliases required by the minimal API

    @property
    def size(self) -> float:
        return self.quantity

    @size.setter
    def size(self, new_size: float) -> None:
        self.quantity = float(new_size)
        # Recompute derived
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    @property
    def entry_price(self) -> float:
        return self.average_price

    @entry_price.setter
    def entry_price(self, new_entry: float) -> None:
        self.average_price = float(new_entry)
        # Recompute derived
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size

    @property
    def current_price(self) -> float:
        return self.market_price if self.market_price is not None else self.average_price

    @current_price.setter
    def current_price(self, new_price: float) -> None:
        self.market_price = float(new_price)
        # Recompute derived
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        self.last_updated = datetime.now()

    # Minimal API

    @property
    def value(self) -> float:
        return self.size * self.current_price

    def update_price(self, new_price: float) -> None:
        """Update the current price; alias of update_market_price(new_price)."""
        self.update_market_price(new_price)

    # Existing API preserved

    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        if self.market_price is None:
            return 0.0
        return self.quantity * self.market_price

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == 0

    def update_market_price(self, new_price: float) -> None:
        """Update market price and recalculate unrealized P&L."""
        self.market_price = float(new_price)
        # Derived and timestamp
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        self.last_updated = datetime.now()

    def add_realized_pnl(self, pnl: float) -> None:
        """Add realized P&L to the position."""
        self.realized_pnl += float(pnl)
        self.last_updated = datetime.now()

    def update_quantity(
        self, new_quantity: float, new_average_price: Optional[float] = None
    ) -> None:
        """Update position quantity and optionally average price."""
        self.quantity = float(new_quantity)
        if new_average_price is not None:
            self.average_price = float(new_average_price)
        # Recompute derived
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.size
        self.last_updated = datetime.now()

    def close(self, exit_price: float, exit_time: Optional[datetime] = None) -> None:
        """Mark position as closed at the given exit price/time."""
        # Update current price and derived fields
        self.update_market_price(exit_price)
        # Record exit time
        self.exit_time = exit_time or datetime.now()
