"""Position tracking utilities for EMP order management.

This module provides a stateful :class:`PositionTracker` capable of ingesting
fills, computing realised/unrealised PnL under FIFO or LIFO accounting, and
reporting account level exposures.  The implementation is intentionally
side-effect free (apart from internal state mutation) so it can be embedded in
services or exercised during dry runs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, Literal, Mapping, MutableMapping, Optional, Tuple

PnLMode = Literal["fifo", "lifo"]

__all__ = [
    "PnLMode",
    "PositionLot",
    "PositionSnapshot",
    "ReconciliationDifference",
    "ReconciliationReport",
    "PositionTracker",
]


_EPSILON = 1e-12


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PositionLot:
    """A single open lot in the inventory."""

    quantity: float
    price: float
    timestamp: datetime


@dataclass(slots=True)
class PositionSnapshot:
    """Summary of the tracked state for a symbol and account."""

    symbol: str
    account: str
    net_quantity: float
    long_quantity: float
    short_quantity: float
    market_price: Optional[float]
    average_long_price: Optional[float]
    average_short_price: Optional[float]
    realized_pnl: float
    unrealized_pnl: Optional[float]
    exposure: Optional[float]

    @property
    def market_value(self) -> Optional[float]:
        """Return the market value of the net position if a price is known."""

        if self.market_price is None:
            return None
        return self.net_quantity * self.market_price


@dataclass(slots=True)
class ReconciliationDifference:
    """Represents a delta between tracked and broker reported quantity."""

    symbol: str
    tracker_quantity: float
    broker_quantity: float
    difference: float


@dataclass(slots=True)
class ReconciliationReport:
    """Container describing the outcome of a reconciliation pass."""

    timestamp: datetime
    account: str
    differences: Tuple[ReconciliationDifference, ...]

    def has_discrepancies(self) -> bool:
        return bool(self.differences)


class _TrackedPosition:
    """Internal structure keeping lots and realised PnL for a symbol."""

    __slots__ = (
        "long_lots",
        "short_lots",
        "realized_pnl",
        "fees",
        "last_price",
        "last_timestamp",
    )

    def __init__(self) -> None:
        self.long_lots: Deque[PositionLot] = deque()
        self.short_lots: Deque[PositionLot] = deque()
        self.realized_pnl: float = 0.0
        self.fees: float = 0.0
        self.last_price: Optional[float] = None
        self.last_timestamp: Optional[datetime] = None

    # --- mutation helpers -------------------------------------------------
    def add_long(self, quantity: float, price: float, timestamp: datetime) -> None:
        if quantity <= _EPSILON:
            return
        self.long_lots.append(PositionLot(quantity=quantity, price=price, timestamp=timestamp))

    def add_short(self, quantity: float, price: float, timestamp: datetime) -> None:
        if quantity <= _EPSILON:
            return
        self.short_lots.append(PositionLot(quantity=quantity, price=price, timestamp=timestamp))

    def close_longs(self, quantity: float, price: float, mode: PnLMode) -> Tuple[float, float]:
        realized = 0.0
        remaining = quantity
        while remaining > _EPSILON and self.long_lots:
            lot = self.long_lots[0] if mode == "fifo" else self.long_lots[-1]
            matched = min(remaining, lot.quantity)
            realized += (price - lot.price) * matched
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= _EPSILON:
                if mode == "fifo":
                    self.long_lots.popleft()
                else:
                    self.long_lots.pop()
        return realized, remaining

    def close_shorts(self, quantity: float, price: float, mode: PnLMode) -> Tuple[float, float]:
        realized = 0.0
        remaining = quantity
        while remaining > _EPSILON and self.short_lots:
            lot = self.short_lots[0] if mode == "fifo" else self.short_lots[-1]
            matched = min(remaining, lot.quantity)
            realized += (lot.price - price) * matched
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= _EPSILON:
                if mode == "fifo":
                    self.short_lots.popleft()
                else:
                    self.short_lots.pop()
        return realized, remaining

    # --- measurement helpers ---------------------------------------------
    def total_long_quantity(self) -> float:
        return sum(lot.quantity for lot in self.long_lots)

    def total_short_quantity(self) -> float:
        return sum(lot.quantity for lot in self.short_lots)

    def net_quantity(self) -> float:
        return self.total_long_quantity() - self.total_short_quantity()

    def average_long_price(self) -> Optional[float]:
        total_qty = self.total_long_quantity()
        if total_qty <= _EPSILON:
            return None
        total_cost = sum(lot.price * lot.quantity for lot in self.long_lots)
        return total_cost / total_qty

    def average_short_price(self) -> Optional[float]:
        total_qty = self.total_short_quantity()
        if total_qty <= _EPSILON:
            return None
        total_cost = sum(lot.price * lot.quantity for lot in self.short_lots)
        return total_cost / total_qty

    def unrealized_pnl(self, market_price: Optional[float]) -> Optional[float]:
        if market_price is None:
            return None
        long_component = sum((market_price - lot.price) * lot.quantity for lot in self.long_lots)
        short_component = sum((lot.price - market_price) * lot.quantity for lot in self.short_lots)
        return long_component + short_component

    def notional_exposure(self, market_price: Optional[float]) -> Optional[float]:
        if market_price is None:
            return None
        return abs(self.net_quantity()) * market_price


class PositionTracker:
    """Track positions, PnL, and exposures at the account level."""

    def __init__(
        self,
        *,
        pnl_mode: PnLMode = "fifo",
        default_account: str = "PRIMARY",
    ) -> None:
        if pnl_mode not in ("fifo", "lifo"):
            raise ValueError("pnl_mode must be either 'fifo' or 'lifo'")
        self._mode = pnl_mode
        self._default_account = default_account
        self._positions: Dict[Tuple[str, str], _TrackedPosition] = {}
        self._marks: MutableMapping[str, tuple[float, datetime]] = {}

    # --- internal utilities ----------------------------------------------
    def _resolve_account(self, account: Optional[str]) -> str:
        return account or self._default_account

    def _get_position(self, account: str, symbol: str) -> _TrackedPosition:
        key = (account, symbol)
        position = self._positions.get(key)
        if position is None:
            position = _TrackedPosition()
            self._positions[key] = position
        return position

    def _peek_position(self, account: str, symbol: str) -> Optional[_TrackedPosition]:
        return self._positions.get((account, symbol))

    def _resolve_mark_price(self, symbol: str, position: Optional[_TrackedPosition]) -> Optional[float]:
        if symbol in self._marks:
            return self._marks[symbol][0]
        if position is not None:
            return position.last_price
        return None

    # --- public API -------------------------------------------------------
    def record_fill(
        self,
        symbol: str,
        quantity: float,
        price: float,
        *,
        account: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        fees: float = 0.0,
    ) -> PositionSnapshot:
        """Register an executed fill and update tracked state.

        Args:
            symbol: Instrument identifier.
            quantity: Signed quantity (positive for buy, negative for sell).
            price: Execution price.
            account: Optional account identifier (defaults to tracker default).
            timestamp: Timestamp of the fill (UTC assumed if naive).
            fees: Optional transaction fees applied to the fill.
        """

        if abs(quantity) <= _EPSILON:
            raise ValueError("Fill quantity must be non-zero")

        account_id = self._resolve_account(account)
        fill_time = timestamp or _utc_now()
        if fill_time.tzinfo is None:
            fill_time = fill_time.replace(tzinfo=timezone.utc)

        position = self._get_position(account_id, symbol)
        position.last_price = price
        position.last_timestamp = fill_time

        abs_quantity = abs(quantity)
        realized_pnl = 0.0
        if quantity > 0:
            realized_delta, remaining = position.close_shorts(abs_quantity, price, self._mode)
            realized_pnl += realized_delta
            if remaining > _EPSILON:
                position.add_long(remaining, price, fill_time)
        else:
            realized_delta, remaining = position.close_longs(abs_quantity, price, self._mode)
            realized_pnl += realized_delta
            if remaining > _EPSILON:
                position.add_short(remaining, price, fill_time)

        position.realized_pnl += realized_pnl - fees
        if fees:
            position.fees += fees

        return self.get_position_snapshot(symbol, account=account_id)

    def update_mark_price(
        self,
        symbol: str,
        price: float,
        *,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update the latest observed market price for a symbol."""

        mark_time = timestamp or _utc_now()
        if mark_time.tzinfo is None:
            mark_time = mark_time.replace(tzinfo=timezone.utc)
        self._marks[symbol] = (price, mark_time)

    def get_position_snapshot(
        self,
        symbol: str,
        *,
        account: Optional[str] = None,
    ) -> PositionSnapshot:
        account_id = self._resolve_account(account)
        position = self._peek_position(account_id, symbol)
        mark_price = self._resolve_mark_price(symbol, position)

        if position is None:
            return PositionSnapshot(
                symbol=symbol,
                account=account_id,
                net_quantity=0.0,
                long_quantity=0.0,
                short_quantity=0.0,
                market_price=mark_price,
                average_long_price=None,
                average_short_price=None,
                realized_pnl=0.0,
                unrealized_pnl=None if mark_price is None else 0.0,
                exposure=None if mark_price is None else 0.0,
            )

        long_qty = position.total_long_quantity()
        short_qty = position.total_short_quantity()
        net_qty = long_qty - short_qty
        unrealized = position.unrealized_pnl(mark_price)
        exposure = position.notional_exposure(mark_price)

        return PositionSnapshot(
            symbol=symbol,
            account=account_id,
            net_quantity=net_qty,
            long_quantity=long_qty,
            short_quantity=short_qty,
            market_price=mark_price,
            average_long_price=position.average_long_price(),
            average_short_price=position.average_short_price(),
            realized_pnl=position.realized_pnl,
            unrealized_pnl=unrealized,
            exposure=exposure,
        )

    def iter_positions(
        self,
        *,
        account: Optional[str] = None,
    ) -> Iterable[PositionSnapshot]:
        """Yield snapshots for all tracked symbols, optionally for one account."""

        if account is None:
            items = self._positions.items()
        else:
            account_id = self._resolve_account(account)
            items = ((key, value) for key, value in self._positions.items() if key[0] == account_id)

        for (account_id, symbol), position in items:
            mark_price = self._resolve_mark_price(symbol, position)
            long_qty = position.total_long_quantity()
            short_qty = position.total_short_quantity()
            net_qty = long_qty - short_qty
            yield PositionSnapshot(
                symbol=symbol,
                account=account_id,
                net_quantity=net_qty,
                long_quantity=long_qty,
                short_quantity=short_qty,
                market_price=mark_price,
                average_long_price=position.average_long_price(),
                average_short_price=position.average_short_price(),
                realized_pnl=position.realized_pnl,
                unrealized_pnl=position.unrealized_pnl(mark_price),
                exposure=position.notional_exposure(mark_price),
            )

    def total_exposure(self, *, account: Optional[str] = None) -> float:
        """Aggregate notional exposure across all tracked positions."""

        total = 0.0
        for snapshot in self.iter_positions(account=account):
            if snapshot.exposure is not None:
                total += snapshot.exposure
        return total

    def generate_reconciliation_report(
        self,
        broker_positions: Mapping[str, float],
        *,
        account: Optional[str] = None,
        tolerance: float = 1e-6,
    ) -> ReconciliationReport:
        """Compare tracked quantities against broker provided balances."""

        account_id = self._resolve_account(account)
        timestamp = _utc_now()
        tracked_symbols = {
            symbol
            for (acct, symbol) in self._positions
            if acct == account_id
        }
        symbols = tracked_symbols.union(broker_positions.keys())
        differences: list[ReconciliationDifference] = []

        for symbol in sorted(symbols):
            snapshot = self.get_position_snapshot(symbol, account=account_id)
            tracker_qty = snapshot.net_quantity
            broker_qty = float(broker_positions.get(symbol, 0.0))
            diff = tracker_qty - broker_qty
            if abs(diff) > tolerance:
                differences.append(
                    ReconciliationDifference(
                        symbol=symbol,
                        tracker_quantity=tracker_qty,
                        broker_quantity=broker_qty,
                        difference=diff,
                    )
                )

        return ReconciliationReport(
            timestamp=timestamp,
            account=account_id,
            differences=tuple(differences),
        )

    def accounts(self) -> Tuple[str, ...]:
        """Return a tuple of accounts with tracked activity."""

        return tuple(sorted({account for account, _ in self._positions}))

    def reset(self) -> None:
        """Clear all tracked state."""

        self._positions.clear()
        self._marks.clear()
