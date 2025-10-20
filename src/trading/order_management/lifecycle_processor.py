"""Co-ordinates FIX broker callbacks with the order state machine."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, Optional

from .event_journal import OrderEventJournal
from .order_state_machine import (
    OrderExecutionEvent,
    OrderLifecycleSnapshot,
    OrderMetadata,
    OrderState,
    OrderStateError,
    OrderStateMachine,
)
from .position_tracker import PositionTracker
from .monitoring import OrderLatencyMonitor

try:  # pragma: no cover - optional import for type checking only
    from ..integration.fix_broker_interface import FIXBrokerInterface
except Exception:  # pragma: no cover - avoid import errors at runtime
    FIXBrokerInterface = object  # type: ignore

from src.operational.structured_logging import get_logger, order_logging_context

__all__ = ["OrderLifecycleProcessor"]

logger = get_logger(__name__)


class OrderLifecycleProcessor:
    """Glue code between the FIX broker interface and the state machine."""

    _EVENT_TYPES = ("acknowledged", "partial_fill", "filled", "cancelled", "rejected")

    def __init__(
        self,
        *,
        journal: Optional[OrderEventJournal] = None,
        state_machine: Optional[OrderStateMachine] = None,
        position_tracker: PositionTracker | None = None,
        account_resolver: Callable[[OrderMetadata], str | None] | None = None,
        latency_monitor: OrderLatencyMonitor | None = None,
    ) -> None:
        self._journal = journal
        self._state_machine = state_machine or OrderStateMachine()
        self._attached_broker: Optional[FIXBrokerInterface] = None
        self._position_tracker = position_tracker
        self._account_resolver = account_resolver
        self._latency_monitor = latency_monitor
        self._exec_dedupe: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._exec_dedupe_capacity = 4096

    # ------------------------------------------------------------------
    @property
    def state_machine(self) -> OrderStateMachine:
        return self._state_machine

    @property
    def journal(self) -> OrderEventJournal | None:
        """Return the backing journal used for lifecycle persistence."""

        return self._journal

    @property
    def position_tracker(self) -> PositionTracker | None:
        """Expose the shared position tracker, if configured."""

        return self._position_tracker

    def register_order(self, metadata: OrderMetadata) -> OrderState:
        """Register a new order with the underlying state machine."""

        return self._state_machine.register_order(metadata)

    # ------------------------------------------------------------------
    def _resolve_account(self, metadata: OrderMetadata) -> str | None:
        if metadata.account:
            return metadata.account
        if self._account_resolver is not None:
            return self._account_resolver(metadata)
        return None

    @staticmethod
    def _normalise_exec_id(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            try:
                value = value.decode()
            except Exception:
                return None
        text = str(value).strip()
        return text or None

    def _extract_exec_id(self, payload: dict) -> str | None:
        for key in ("exec_id", "execId", "ExecID", "execID", "ExecId"):
            if key in payload:
                return self._normalise_exec_id(payload.get(key))
        return None

    def _has_seen_exec(self, order_id: str, exec_id: str) -> bool:
        return (order_id, exec_id) in self._exec_dedupe

    def _remember_exec(self, order_id: str, exec_id: str) -> None:
        key = (order_id, exec_id)
        cache = self._exec_dedupe
        cache[key] = None
        if len(cache) > self._exec_dedupe_capacity:
            cache.popitem(last=False)

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
                logger.debug("lifecycle_listener_detach_failed", event_type=event_type)
        self._attached_broker = None

    # ------------------------------------------------------------------
    def _handle_broker_event(self, order_id: str, payload: dict) -> None:
        try:
            self.handle_broker_payload(order_id, payload)
        except OrderStateError as exc:
            if self._journal is not None:
                self._journal.append_error(payload, reason=str(exc))
            with order_logging_context(order_id):
                logger.warning("order_event_rejected", error=str(exc))

    def handle_broker_payload(
        self, order_id: str, payload: dict
    ) -> OrderLifecycleSnapshot:
        """Convert a broker payload into an execution event and apply it."""

        event = OrderExecutionEvent.from_broker_payload(order_id, payload)
        exec_id = self._extract_exec_id(payload)
        if exec_id and self._has_seen_exec(order_id, exec_id):
            return self._state_machine.snapshot(order_id)
        state = self._state_machine.apply_event(event)
        snapshot = self._state_machine.snapshot(event.order_id)
        self._update_position_tracker(state, event, snapshot)
        if self._latency_monitor is not None:
            self._latency_monitor.record_transition(state, event, snapshot)
        if self._journal is not None:
            self._journal.append(event, snapshot)
        if exec_id:
            self._remember_exec(order_id, exec_id)
        return snapshot

    # ------------------------------------------------------------------
    def get_snapshot(self, order_id: str) -> OrderLifecycleSnapshot:
        return self._state_machine.snapshot(order_id)

    def iter_open_orders(self) -> Iterable[OrderLifecycleSnapshot]:
        return self._state_machine.iter_open_orders()

    def reset(self) -> None:
        self._state_machine.reset()
        self._exec_dedupe.clear()

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

    # ------------------------------------------------------------------
    def _update_position_tracker(
        self,
        state: OrderState,
        event: OrderExecutionEvent,
        snapshot: OrderLifecycleSnapshot,
    ) -> None:
        tracker = self._position_tracker
        if tracker is None:
            return

        if event.event_type not in {"partial_fill", "filled"}:
            if event.last_price is not None:
                tracker.update_mark_price(state.metadata.symbol, float(event.last_price))
            return

        fill_quantity = state.last_fill_quantity
        if fill_quantity <= 0:
            return

        price = event.last_price or state.last_fill_price or snapshot.average_fill_price
        if price is None:
            with order_logging_context(state.metadata.order_id):
                logger.debug("order_fill_without_price_skipped")
            return

        signed_quantity = fill_quantity if state.metadata.side == "BUY" else -fill_quantity
        tracker.record_fill(
            state.metadata.symbol,
            signed_quantity,
            float(price),
            account=self._resolve_account(state.metadata),
        )

        tracker.update_mark_price(state.metadata.symbol, float(price))
