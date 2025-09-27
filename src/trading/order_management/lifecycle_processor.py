"""Co-ordinates FIX broker callbacks with the order state machine."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from .event_journal import OrderEventJournal
from .order_state_machine import (
    OrderExecutionEvent,
    OrderLifecycleSnapshot,
    OrderMetadata,
    OrderState,
    OrderStateError,
    OrderStateMachine,
)

try:  # pragma: no cover - optional import for type checking only
    from ..integration.fix_broker_interface import FIXBrokerInterface
except Exception:  # pragma: no cover - avoid import errors at runtime
    FIXBrokerInterface = object  # type: ignore

__all__ = ["OrderLifecycleProcessor"]

logger = logging.getLogger(__name__)


class OrderLifecycleProcessor:
    """Glue code between the FIX broker interface and the state machine."""

    _EVENT_TYPES = ("acknowledged", "partial_fill", "filled", "cancelled", "rejected")

    def __init__(
        self,
        *,
        journal: Optional[OrderEventJournal] = None,
        state_machine: Optional[OrderStateMachine] = None,
    ) -> None:
        self._journal = journal
        self._state_machine = state_machine or OrderStateMachine()
        self._attached_broker: Optional[FIXBrokerInterface] = None

    # ------------------------------------------------------------------
    @property
    def state_machine(self) -> OrderStateMachine:
        return self._state_machine

    def register_order(self, metadata: OrderMetadata) -> OrderState:
        """Register a new order with the underlying state machine."""

        return self._state_machine.register_order(metadata)

    # ------------------------------------------------------------------
    def attach_broker(self, broker: FIXBrokerInterface) -> None:
        """Register callbacks on the broker for lifecycle events."""

        for event_type in self._EVENT_TYPES:
            broker.add_event_listener(event_type, self._handle_broker_event)
        self._attached_broker = broker

    def detach_broker(self) -> None:
        broker = self._attached_broker
        if broker is None:
            return
        for event_type in self._EVENT_TYPES:
            try:
                broker.remove_event_listener(event_type, self._handle_broker_event)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to remove %s listener", event_type)
        self._attached_broker = None

    # ------------------------------------------------------------------
    def _handle_broker_event(self, order_id: str, payload: dict) -> None:
        try:
            self.handle_broker_payload(order_id, payload)
        except OrderStateError as exc:
            if self._journal is not None:
                self._journal.append_error(payload, reason=str(exc))
            logger.warning("Rejected order event for %s: %s", order_id, exc)

    def handle_broker_payload(
        self, order_id: str, payload: dict
    ) -> OrderLifecycleSnapshot:
        """Convert a broker payload into an execution event and apply it."""

        event = OrderExecutionEvent.from_broker_payload(order_id, payload)
        self._state_machine.apply_event(event)
        snapshot = self._state_machine.snapshot(event.order_id)
        if self._journal is not None:
            self._journal.append(event, snapshot)
        return snapshot

    # ------------------------------------------------------------------
    def get_snapshot(self, order_id: str) -> OrderLifecycleSnapshot:
        return self._state_machine.snapshot(order_id)

    def iter_open_orders(self) -> Iterable[OrderLifecycleSnapshot]:
        return self._state_machine.iter_open_orders()

    def reset(self) -> None:
        self._state_machine.reset()

    # Convenience wrappers for integration tests --------------------------------
    def apply_acknowledgement(self, order_id: str, payload: dict[str, object] | None = None) -> OrderLifecycleSnapshot:
        payload = {"exec_type": "0", **(payload or {})}
        return self.handle_broker_payload(order_id, payload)

    def apply_partial_fill(self, order_id: str, payload: dict[str, object] | None = None) -> OrderLifecycleSnapshot:
        payload = {"exec_type": "1", **(payload or {})}
        return self.handle_broker_payload(order_id, payload)

    def apply_fill(self, order_id: str, payload: dict[str, object] | None = None) -> OrderLifecycleSnapshot:
        payload = {"exec_type": "2", **(payload or {})}
        return self.handle_broker_payload(order_id, payload)

    def apply_cancel(self, order_id: str, payload: dict[str, object] | None = None) -> OrderLifecycleSnapshot:
        payload = {"exec_type": "4", **(payload or {})}
        return self.handle_broker_payload(order_id, payload)

    def apply_reject(self, order_id: str, payload: dict[str, object] | None = None) -> OrderLifecycleSnapshot:
        payload = {"exec_type": "8", **(payload or {})}
        return self.handle_broker_payload(order_id, payload)
