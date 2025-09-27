"""State machine utilities for FIX order lifecycle management.

This module provides a focused state machine that mirrors the roadmap
expectation for institutional order handling.  It translates FIX execution
events into deterministic order states, keeps a history of lifecycle events,
and exposes immutable snapshots that downstream components can consume for
monitoring, reconciliation, or telemetry exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "LifecycleStatus",
    "OrderEventType",
    "OrderEvent",
    "OrderLifecycleSnapshot",
    "OrderLifecycle",
    "OrderStateError",
]


class OrderStateError(RuntimeError):
    """Raised when an invalid state transition is attempted."""


class LifecycleStatus(str, Enum):
    """Enumerates the canonical states for an order lifecycle."""

    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderEventType(str, Enum):
    """Lifecycle events that drive state transitions."""

    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class OrderEvent:
    """Represents a single lifecycle event."""

    type: OrderEventType
    timestamp: datetime
    quantity: float = 0.0
    price: Optional[float] = None
    leaves_quantity: Optional[float] = None
    reason: Optional[str] = None
    raw_exec_type: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "quantity": self.quantity,
            "price": self.price,
            "leaves_quantity": self.leaves_quantity,
            "reason": self.reason,
            "raw_exec_type": self.raw_exec_type,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OrderLifecycleSnapshot:
    """Immutable view of the current order lifecycle state."""

    order_id: str
    status: LifecycleStatus
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    created_at: datetime
    updated_at: datetime
    symbol: Optional[str]
    side: Optional[str]
    initial_quantity: Optional[float]
    last_event: Optional[OrderEvent]
    reason: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "initial_quantity": self.initial_quantity,
            "reason": self.reason,
        }
        if self.last_event is not None:
            payload["last_event"] = self.last_event.to_dict()
        return payload


_FIX_EXEC_TYPE_TO_EVENT: dict[str, OrderEventType] = {
    "0": OrderEventType.ACKNOWLEDGED,
    "1": OrderEventType.PARTIAL_FILL,
    "2": OrderEventType.FILL,
    "4": OrderEventType.CANCELLED,
    "8": OrderEventType.REJECTED,
}

_ALLOWED_TRANSITIONS: Mapping[LifecycleStatus, set[LifecycleStatus]] = {
    LifecycleStatus.NEW: {
        LifecycleStatus.ACKNOWLEDGED,
        LifecycleStatus.PARTIALLY_FILLED,
        LifecycleStatus.FILLED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.REJECTED,
    },
    LifecycleStatus.ACKNOWLEDGED: {
        LifecycleStatus.ACKNOWLEDGED,
        LifecycleStatus.PARTIALLY_FILLED,
        LifecycleStatus.FILLED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.REJECTED,
    },
    LifecycleStatus.PARTIALLY_FILLED: {
        LifecycleStatus.PARTIALLY_FILLED,
        LifecycleStatus.FILLED,
        LifecycleStatus.CANCELLED,
    },
    LifecycleStatus.FILLED: set(),
    LifecycleStatus.CANCELLED: set(),
    LifecycleStatus.REJECTED: set(),
}

_FILL_EPSILON = 1e-9


def _ensure_utc(timestamp: Optional[datetime]) -> datetime:
    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return float(value.decode())
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not decode float value: {value!r}") from exc
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not convert value to float: {value!r}") from exc


class OrderLifecycle:
    """Track a single order lifecycle and enforce roadmap transitions."""

    def __init__(
        self,
        order_id: str,
        *,
        quantity: Optional[float] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        self.order_id = order_id
        self._initial_quantity: Optional[float] = float(quantity) if quantity is not None else None
        self.symbol = symbol
        self.side = side
        self._created_at = _ensure_utc(created_at)
        self._updated_at = self._created_at
        self._status = LifecycleStatus.NEW
        self._filled_quantity = 0.0
        self._average_price: Optional[float] = None
        self._last_leaves: Optional[float] = None
        self._last_event: Optional[OrderEvent] = None
        self._reason: Optional[str] = None
        self._events: list[OrderEvent] = []
        self._record_event(OrderEventType.NEW, timestamp=self._created_at)

    @property
    def status(self) -> LifecycleStatus:
        return self._status

    @property
    def initial_quantity(self) -> Optional[float]:
        return self._initial_quantity

    @property
    def filled_quantity(self) -> float:
        return self._filled_quantity

    @property
    def average_price(self) -> Optional[float]:
        return self._average_price

    @property
    def events(self) -> Iterable[OrderEvent]:
        return tuple(self._events)

    def set_initial_quantity(self, quantity: float) -> None:
        if quantity <= 0:
            return
        if self._initial_quantity is None:
            self._initial_quantity = float(quantity)

    def _record_event(
        self,
        event_type: OrderEventType,
        *,
        timestamp: Optional[datetime] = None,
        quantity: float = 0.0,
        price: Optional[float] = None,
        leaves_quantity: Optional[float] = None,
        reason: Optional[str] = None,
        raw_exec_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OrderEvent:
        event = OrderEvent(
            type=event_type,
            timestamp=_ensure_utc(timestamp),
            quantity=quantity,
            price=price,
            leaves_quantity=leaves_quantity,
            reason=reason,
            raw_exec_type=raw_exec_type,
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        self._last_event = event
        self._updated_at = event.timestamp
        self._reason = reason
        if leaves_quantity is not None:
            self._last_leaves = max(leaves_quantity, 0.0)
        return event

    def _validate_transition(self, target_status: LifecycleStatus) -> None:
        if self._status == target_status:
            return
        allowed = _ALLOWED_TRANSITIONS[self._status]
        if target_status not in allowed:
            raise OrderStateError(
                f"Invalid transition from {self._status.value} to {target_status.value}"
            )

    def _update_fill_metrics(
        self,
        quantity: float,
        price: Optional[float],
        leaves_quantity: Optional[float],
    ) -> None:
        if quantity < -_FILL_EPSILON:
            raise OrderStateError("Fill quantity cannot be negative")

        new_total = self._filled_quantity + quantity
        if self._initial_quantity is not None and new_total > self._initial_quantity + _FILL_EPSILON:
            raise OrderStateError(
                "Filled quantity exceeds the initial order quantity"
            )

        if self._initial_quantity is None and leaves_quantity is not None:
            inferred = new_total + max(leaves_quantity, 0.0)
            if inferred > 0:
                self._initial_quantity = inferred

        self._filled_quantity = new_total

        if quantity > _FILL_EPSILON and price is not None:
            if self._average_price is None:
                self._average_price = price
            else:
                total_notional = self._average_price * (new_total - quantity) + price * quantity
                if new_total > _FILL_EPSILON:
                    self._average_price = total_notional / new_total

        if leaves_quantity is not None:
            self._last_leaves = max(leaves_quantity, 0.0)
        elif self._initial_quantity is not None:
            self._last_leaves = max(self._initial_quantity - self._filled_quantity, 0.0)

    def apply_event(
        self,
        event_type: OrderEventType,
        *,
        timestamp: Optional[datetime] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        leaves_quantity: Optional[float] = None,
        reason: Optional[str] = None,
        raw_exec_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OrderLifecycleSnapshot:
        qty = float(quantity or 0.0)
        px = price
        leaves = leaves_quantity

        target_status = {
            OrderEventType.NEW: LifecycleStatus.NEW,
            OrderEventType.ACKNOWLEDGED: LifecycleStatus.ACKNOWLEDGED,
            OrderEventType.PARTIAL_FILL: LifecycleStatus.PARTIALLY_FILLED,
            OrderEventType.FILL: LifecycleStatus.FILLED,
            OrderEventType.CANCELLED: LifecycleStatus.CANCELLED,
            OrderEventType.REJECTED: LifecycleStatus.REJECTED,
        }[event_type]

        if self._status in {LifecycleStatus.FILLED, LifecycleStatus.CANCELLED, LifecycleStatus.REJECTED}:
            raise OrderStateError(f"Order lifecycle already terminal: {self._status.value}")

        if event_type in {OrderEventType.CANCELLED, OrderEventType.REJECTED} and leaves is None:
            leaves = 0.0

        if event_type in {OrderEventType.PARTIAL_FILL, OrderEventType.FILL}:
            self._update_fill_metrics(qty, px, leaves)
            if (
                event_type == OrderEventType.PARTIAL_FILL
                and ((leaves is not None and leaves <= _FILL_EPSILON)
                     or (
                        self._initial_quantity is not None
                        and self._initial_quantity - self._filled_quantity <= _FILL_EPSILON
                    ))
            ):
                target_status = LifecycleStatus.FILLED

        self._validate_transition(target_status)
        self._status = target_status

        event = self._record_event(
            event_type,
            timestamp=timestamp,
            quantity=qty,
            price=px,
            leaves_quantity=leaves,
            reason=reason,
            raw_exec_type=raw_exec_type,
            metadata=metadata,
        )

        return self.snapshot()

    def apply_fix_execution(
        self,
        exec_type: Any,
        *,
        last_qty: Any = None,
        last_px: Any = None,
        leaves_qty: Any = None,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OrderLifecycleSnapshot:
        exec_value = exec_type.decode() if isinstance(exec_type, (bytes, bytearray)) else str(exec_type)
        event_type = _FIX_EXEC_TYPE_TO_EVENT.get(exec_value)
        if event_type is None:
            raise OrderStateError(f"Unsupported FIX ExecType: {exec_type!r}")

        qty = _coerce_float(last_qty) or 0.0
        px = _coerce_float(last_px) if last_px is not None else None
        leaves = _coerce_float(leaves_qty) if leaves_qty is not None else None

        return self.apply_event(
            event_type,
            timestamp=timestamp,
            quantity=qty,
            price=px,
            leaves_quantity=leaves,
            reason=reason,
            raw_exec_type=exec_value,
            metadata=metadata,
        )

    def snapshot(self) -> OrderLifecycleSnapshot:
        remaining = self._last_leaves
        if remaining is None:
            if self._initial_quantity is not None:
                remaining = max(self._initial_quantity - self._filled_quantity, 0.0)
            else:
                remaining = 0.0

        return OrderLifecycleSnapshot(
            order_id=self.order_id,
            status=self._status,
            filled_quantity=self._filled_quantity,
            remaining_quantity=remaining,
            average_price=self._average_price,
            created_at=self._created_at,
            updated_at=self._updated_at,
            symbol=self.symbol,
            side=self.side,
            initial_quantity=self._initial_quantity,
            last_event=self._last_event,
            reason=self._reason,
        )

