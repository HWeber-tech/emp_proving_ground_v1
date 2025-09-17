from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Callable, Dict, Iterable, Optional

from src.core.interfaces import IExecutionEngine
from src.trading.models.order import Order, OrderStatus, OrderType
from src.trading.models.position import Position


@dataclass
class _TrackedOrder:
    """Internal helper that keeps metadata alongside an :class:`Order`."""

    order: Order
    attempts: int = 1
    last_error: str | None = None

    def remaining(self) -> float:
        return max(self.order.quantity - self.order.filled_quantity, 0.0)

    def snapshot(self) -> dict[str, object]:
        return {
            "order_id": self.order.order_id,
            "status": self.order.status.value,
            "attempts": self.attempts,
            "filled_quantity": self.order.filled_quantity,
            "remaining_quantity": self.remaining(),
            "last_error": self.last_error,
        }


class ExecutionEngine(IExecutionEngine):
    """In-memory execution adapter with deterministic, test-friendly behaviour."""

    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._id_sequence = count(1)
        self._id_factory = id_factory or self._default_id_factory
        self._clock = clock or datetime.utcnow
        self._orders: Dict[str, _TrackedOrder] = {}
        self._positions: Dict[str, Position] = {}

    def _default_id_factory(self) -> str:
        return f"ORD-{next(self._id_sequence)}"

    def _normalise_order_type(self, order_type: OrderType | str | None) -> OrderType:
        if isinstance(order_type, OrderType):
            return order_type
        if isinstance(order_type, str):
            key = order_type.upper()
            if key in OrderType.__members__:
                return OrderType[key]
        return OrderType.MARKET

    def _require_order(self, order_id: str) -> _TrackedOrder:
        try:
            return self._orders[order_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown order_id: {order_id}") from exc

    def _resolve_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol, quantity=0.0, average_price=0.0)
        return self._positions[symbol]

    async def send_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        *,
        order_type: OrderType | str | None = None,
        stop_price: float | None = None,
    ) -> str:
        side_normalised = side.upper()
        if side_normalised not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side '{side}'")
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")

        order_id = self._id_factory()
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side_normalised,  # type: ignore[arg-type]
            quantity=float(quantity),
            order_type=self._normalise_order_type(order_type),
            price=price,
            stop_price=stop_price,
            created_at=self._clock(),
            updated_at=self._clock(),
        )
        self._orders[order_id] = _TrackedOrder(order=order)
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        tracker = self._orders.get(order_id)
        if tracker is None:
            return False
        if tracker.order.status == OrderStatus.FILLED:
            return False
        tracker.order.update_status(OrderStatus.CANCELLED)
        return True

    async def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_order(self, order_id: str) -> Order:
        return self._require_order(order_id).order

    def iter_orders(self) -> Iterable[Order]:
        for tracker in self._orders.values():
            yield tracker.order

    def mark_rejected(self, order_id: str, reason: str) -> None:
        tracker = self._require_order(order_id)
        tracker.last_error = reason
        tracker.order.update_status(OrderStatus.REJECTED)

    def retry_order(self, order_id: str, *, note: str | None = None) -> bool:
        tracker = self._orders.get(order_id)
        if tracker is None or tracker.order.status == OrderStatus.FILLED:
            return False
        tracker.attempts += 1
        if note is not None:
            tracker.last_error = note
        tracker.order.update_status(OrderStatus.PENDING)
        return True

    def record_fill(self, order_id: str, quantity: float, price: float) -> None:
        tracker = self._require_order(order_id)
        order = tracker.order
        if order.status == OrderStatus.CANCELLED:
            raise ValueError("Cannot fill a cancelled order")
        if quantity <= 0:
            return

        remaining_before = max(order.quantity - order.filled_quantity, 0.0)
        fill_quantity = min(quantity, remaining_before)
        if fill_quantity <= 0:
            return

        order.add_fill(fill_quantity, price)
        self._apply_fill_to_position(order, fill_quantity, price)

    def _apply_fill_to_position(self, order: Order, quantity: float, price: float) -> None:
        position = self._resolve_position(order.symbol)
        position.update_market_price(price)

        if order.side == "BUY":
            new_quantity = position.quantity + quantity
            if new_quantity <= 0:
                position.update_quantity(0.0, position.average_price)
                return
            previous_value = position.quantity * position.average_price
            incoming_value = quantity * price
            average_price = (previous_value + incoming_value) / new_quantity
            position.update_quantity(new_quantity, average_price)
        else:  # SELL
            position.add_realized_pnl((price - position.average_price) * quantity)
            position.update_quantity(position.quantity - quantity, position.average_price)

    def reconcile(self) -> dict[str, object]:
        open_orders: list[dict[str, object]] = []
        filled_orders: list[dict[str, object]] = []
        cancelled_orders: list[dict[str, object]] = []

        for tracker in self._orders.values():
            bucket = open_orders
            if tracker.order.status == OrderStatus.FILLED:
                bucket = filled_orders
            elif tracker.order.status == OrderStatus.CANCELLED:
                bucket = cancelled_orders
            bucket.append(tracker.snapshot())

        positions_snapshot: dict[str, dict[str, float]] = {}
        for symbol, position in self._positions.items():
            if abs(position.quantity) <= 1e-9 and abs(position.realized_pnl) <= 1e-9:
                continue
            positions_snapshot[symbol] = {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "realized_pnl": position.realized_pnl,
            }

        return {
            "open_orders": open_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "positions": positions_snapshot,
        }


__all__ = ["ExecutionEngine"]
