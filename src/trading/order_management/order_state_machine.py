"""State machine for managing FIX order lifecycles.

This module provides a light-weight but strictly validated representation of an
order lifecycle so downstream components (risk, telemetry, reconciliation)
receive consistent state transitions.  The implementation intentionally mirrors
the roadmap requirement for deterministic New → Acknowledged →
Partially Filled → (Filled|Cancelled|Rejected) flows with quantity and price
parity across events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Iterable, Literal, Optional

__all__ = [
    "OrderStatus",
    "OrderStateError",
    "OrderMetadata",
    "OrderExecutionEvent",
    "OrderState",
    "OrderLifecycleSnapshot",
    "OrderStateMachine",
]


def _utc_now() -> datetime:
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    return now


class OrderStatus(str, Enum):
    """Enumerates the supported lifecycle states for an order."""

    PENDING_NEW = "PENDING_NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


TERMINAL_STATES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
}


ExecutionEventType = Literal[
    "acknowledged",
    "partial_fill",
    "filled",
    "cancelled",
    "rejected",
]


class OrderStateError(RuntimeError):
    """Raised when an invalid transition or payload is applied."""


@dataclass(slots=True, frozen=True)
class OrderMetadata:
    """Static order attributes supplied at creation time."""

    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(slots=True, frozen=True)
class OrderExecutionEvent:
    """Execution report normalised into typed attributes."""

    order_id: str
    event_type: ExecutionEventType
    exec_type: str
    last_quantity: Optional[float] = None
    last_price: Optional[float] = None
    cumulative_quantity: Optional[float] = None
    leaves_quantity: Optional[float] = None
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.last_quantity is not None and self.last_quantity < 0:
            raise ValueError("last_quantity cannot be negative")
        if self.cumulative_quantity is not None and self.cumulative_quantity < 0:
            raise ValueError("cumulative_quantity cannot be negative")
        if self.leaves_quantity is not None and self.leaves_quantity < 0:
            raise ValueError("leaves_quantity cannot be negative")
        if self.event_type in {"partial_fill", "filled"}:
            if self.last_quantity is None and self.cumulative_quantity is None:
                raise ValueError(
                    "fill events require last_quantity or cumulative_quantity"
                )

    @classmethod
    def from_broker_payload(cls, order_id: str, payload: Dict[str, object]) -> "OrderExecutionEvent":
        """Create an event from a broker update payload."""

        exec_type = str(payload.get("exec_type", ""))
        event_map: Dict[str, ExecutionEventType] = {
            "0": "acknowledged",
            "1": "partial_fill",
            "2": "filled",
            "4": "cancelled",
            "8": "rejected",
        }
        try:
            event_type = event_map[exec_type]
        except KeyError as exc:  # pragma: no cover - defensive
            raise OrderStateError(f"Unsupported exec_type: {exec_type!r}") from exc

        timestamp = payload.get("timestamp")
        if isinstance(timestamp, datetime):
            ts = timestamp
        else:
            ts = _utc_now()

        def _coerce_float(value: object | None) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return None

        return cls(
            order_id=order_id,
            event_type=event_type,
            exec_type=exec_type,
            last_quantity=_coerce_float(payload.get("last_qty"))
            or _coerce_float(payload.get("last_quantity")),
            last_price=_coerce_float(payload.get("last_px"))
            or _coerce_float(payload.get("last_price")),
            cumulative_quantity=_coerce_float(payload.get("cum_qty"))
            or _coerce_float(payload.get("cumulative_qty")),
            leaves_quantity=_coerce_float(payload.get("leaves_qty")),
            timestamp=ts,
        )


@dataclass(slots=True)
class OrderState:
    """Mutable state tracked for an order."""

    metadata: OrderMetadata
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    last_event: Optional[ExecutionEventType] = None
    last_update: datetime = field(default_factory=_utc_now)
    events: list[OrderExecutionEvent] = field(default_factory=list)

    @property
    def remaining_quantity(self) -> float:
        return max(self.metadata.quantity - self.filled_quantity, 0.0)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES


@dataclass(slots=True, frozen=True)
class OrderLifecycleSnapshot:
    """Read-only representation of an order state."""

    order_id: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    average_fill_price: Optional[float]
    last_event: Optional[ExecutionEventType]
    last_update: datetime


class OrderStateMachine:
    """Manage order states and validate lifecycle transitions."""

    _quantity_epsilon = 1e-9

    def __init__(self) -> None:
        self._orders: Dict[str, OrderState] = {}

    # -- registration -----------------------------------------------------
    def register_order(self, metadata: OrderMetadata) -> OrderState:
        if metadata.order_id in self._orders:
            raise OrderStateError(f"Order {metadata.order_id} already registered")
        state = OrderState(metadata=metadata)
        self._orders[metadata.order_id] = state
        return state

    def has_order(self, order_id: str) -> bool:
        return order_id in self._orders

    # -- event ingestion --------------------------------------------------
    def apply_event(self, event: OrderExecutionEvent) -> OrderState:
        if event.order_id not in self._orders:
            raise OrderStateError(f"Unknown order {event.order_id}")

        state = self._orders[event.order_id]
        if state.is_terminal and event.event_type != state.last_event:
            raise OrderStateError(
                f"Order {event.order_id} is terminal ({state.status}); event {event.event_type} not allowed"
            )

        if event.event_type == "acknowledged":
            self._apply_ack(state, event)
        elif event.event_type in {"partial_fill", "filled"}:
            self._apply_fill(state, event)
        elif event.event_type == "cancelled":
            self._apply_cancel(state, event)
        elif event.event_type == "rejected":
            self._apply_reject(state, event)
        else:  # pragma: no cover - should be unreachable
            raise OrderStateError(f"Unsupported event type: {event.event_type}")

        state.events.append(event)
        state.last_event = event.event_type
        state.last_update = event.timestamp
        return state

    # -- event handlers ---------------------------------------------------
    def _apply_ack(self, state: OrderState, event: OrderExecutionEvent) -> None:
        if state.status not in {OrderStatus.PENDING_NEW, OrderStatus.ACKNOWLEDGED}:
            raise OrderStateError(
                f"Cannot acknowledge order in state {state.status}"
            )
        state.status = OrderStatus.ACKNOWLEDGED

    def _apply_fill(self, state: OrderState, event: OrderExecutionEvent) -> None:
        if state.status not in {
            OrderStatus.PENDING_NEW,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise OrderStateError(
                f"Cannot apply fill to order in state {state.status}"
            )

        increment = 0.0
        previous_filled = state.filled_quantity

        if event.cumulative_quantity is not None:
            if event.cumulative_quantity + self._quantity_epsilon < previous_filled:
                raise OrderStateError(
                    "Cumulative quantity moved backwards for order " f"{state.metadata.order_id}"
                )
            increment = max(event.cumulative_quantity - previous_filled, 0.0)
            state.filled_quantity = event.cumulative_quantity
        else:
            increment = float(event.last_quantity or 0.0)
            if increment < 0:
                raise OrderStateError("Fill increment cannot be negative")
            state.filled_quantity = previous_filled + increment

        if state.filled_quantity - state.metadata.quantity > self._quantity_epsilon:
            raise OrderStateError(
                f"Order {state.metadata.order_id} overfilled: {state.filled_quantity} > {state.metadata.quantity}"
            )

        if increment > 0 and event.last_price is not None:
            notional_before = previous_filled * (state.average_fill_price or 0.0)
            notional_delta = increment * event.last_price
            new_total = previous_filled + increment
            if new_total <= self._quantity_epsilon:
                state.average_fill_price = event.last_price
            else:
                state.average_fill_price = (notional_before + notional_delta) / new_total

        remaining = state.remaining_quantity
        if remaining <= self._quantity_epsilon:
            state.status = OrderStatus.FILLED
            state.filled_quantity = min(state.metadata.quantity, state.filled_quantity)
        else:
            state.status = OrderStatus.PARTIALLY_FILLED

    def _apply_cancel(self, state: OrderState, event: OrderExecutionEvent) -> None:
        if state.status in TERMINAL_STATES:
            if state.status != OrderStatus.CANCELLED:
                raise OrderStateError(
                    f"Order already terminal as {state.status}, cannot cancel"
                )
            return
        if state.status not in {
            OrderStatus.PENDING_NEW,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise OrderStateError(
                f"Cannot cancel order in state {state.status}"
            )
        state.status = OrderStatus.CANCELLED

    def _apply_reject(self, state: OrderState, event: OrderExecutionEvent) -> None:
        if state.status in TERMINAL_STATES and state.status != OrderStatus.REJECTED:
            raise OrderStateError(
                f"Order already terminal as {state.status}, cannot reject"
            )
        if state.status not in {OrderStatus.PENDING_NEW, OrderStatus.ACKNOWLEDGED}:
            raise OrderStateError(
                f"Cannot reject order in state {state.status}"
            )
        state.status = OrderStatus.REJECTED

    # -- inspection -------------------------------------------------------
    def snapshot(self, order_id: str) -> OrderLifecycleSnapshot:
        state = self._orders.get(order_id)
        if state is None:
            raise OrderStateError(f"Unknown order {order_id}")
        return OrderLifecycleSnapshot(
            order_id=order_id,
            status=state.status,
            filled_quantity=state.filled_quantity,
            remaining_quantity=state.remaining_quantity,
            average_fill_price=state.average_fill_price,
            last_event=state.last_event,
            last_update=state.last_update,
        )

    def iter_open_orders(self) -> Iterable[OrderLifecycleSnapshot]:
        for order_id, state in self._orders.items():
            if not state.is_terminal:
                yield self.snapshot(order_id)

    def reset(self) -> None:
        self._orders.clear()
