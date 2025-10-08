"""
FIX Broker Interface for IC Markets
Provides integration between FIX protocol and trading system
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime

# Add typing for callbacks and awaitables
from collections import defaultdict
from decimal import Decimal
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
)

import simplefix

from src.operational.structured_logging import get_logger, order_logging_context
from src.trading.execution._risk_context import (
    RiskContextProvider,
    capture_risk_context,
    describe_risk_context,
)
from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    TradingRiskInterface,
    merge_risk_references,
)

logger = get_logger(__name__)


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]
OrderEventCallback = Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]


_MANUAL_FIX_RISK_RUNBOOK = "docs/operations/runbooks/manual_fix_order_risk_block.md"

ORDER_EVENT_TYPES = {
    "order_update",
    "acknowledged",
    "partial_fill",
    "filled",
    "cancelled",
    "rejected",
    "cancel_rejected",
}


class SupportsRiskGateway(Protocol):
    """Subset of the trading risk gateway used by the FIX adapter."""

    async def validate_trade_intent(
        self, intent: Any, portfolio_state: Mapping[str, Any] | None
    ) -> Any | None:
        ...

    def get_last_decision(self) -> Mapping[str, Any] | None: ...

    def get_last_policy_snapshot(self) -> Any | None: ...

    def get_risk_limits(self) -> Mapping[str, Any]: ...


class FIXBrokerInterface:
    """Interface between FIX protocol and trading system.

    This adapter exposes a minimal API to place/cancel orders and tracks
    lightweight in-memory order state. It can optionally notify registered
    listeners when execution reports arrive so higher-level components
    (e.g., order lifecycle/position trackers) can react to updates.
    """

    def __init__(
        self,
        event_bus: Any,
        trade_queue: Any,
        fix_initiator: Any,
        *,
        task_factory: TaskFactory | None = None,
        risk_gateway: SupportsRiskGateway | None = None,
        portfolio_state_provider: Callable[[str], Mapping[str, Any]] | None = None,
        risk_event_topic: str = "telemetry.risk.intent_rejected",
        risk_interface_provider: Callable[[], Any] | None = None,
        risk_context_provider: RiskContextProvider | None = None,
    ) -> None:
        """
        Initialize FIX broker interface.

        Args:
            event_bus: Event bus for system communication
            trade_queue: Queue for trade messages
            fix_initiator: FIX initiator for sending orders
        """
        self.event_bus = event_bus
        self.trade_queue = trade_queue
        self.fix_initiator = fix_initiator
        self.running = False
        self.orders: dict[str, dict[str, object]] = {}
        self._order_update_listeners: list[OrderEventCallback] = []  # deprecated path
        self._event_callbacks: dict[str, list[OrderEventCallback]] = defaultdict(list)
        self._trade_task: asyncio.Task[Any] | None = None
        self._task_factory = task_factory
        self._risk_gateway = risk_gateway
        self._portfolio_state_provider = portfolio_state_provider
        self._risk_event_topic = risk_event_topic
        self._risk_interface_provider: Callable[[], Any] | None = None
        if risk_interface_provider is not None:
            self.set_risk_interface_provider(risk_interface_provider)
        self._risk_context_provider: RiskContextProvider | None = None
        self._last_risk_metadata: dict[str, object] | None = None
        self._last_risk_error: dict[str, object] | None = None
        if risk_context_provider is not None:
            self.set_risk_context_provider(risk_context_provider)

    async def start(self) -> None:
        """Start the broker interface."""
        if self.running:
            return

        self.running = True
        logger.info("fix_broker_started")

        # Start message processing
        self._trade_task = self._spawn_task(
            self._process_trade_messages(),
            name="fix-broker-trade-feed",
        )

    def set_risk_interface_provider(
        self, provider: Callable[[], Any] | None
    ) -> None:
        """Register (or clear) the callable that exposes trading risk state."""

        if provider is not None and not callable(provider):
            raise TypeError("risk interface provider must be callable or None")
        self._risk_interface_provider = provider

    def set_risk_context_provider(self, provider: RiskContextProvider | None) -> None:
        """Install or clear the callable used to capture deterministic risk metadata."""

        if provider is not None and not callable(provider):
            raise TypeError("risk context provider must be callable or None")
        self._risk_context_provider = provider

    def _capture_risk_context(self) -> None:
        metadata, error = capture_risk_context(self._risk_context_provider)
        self._last_risk_metadata = metadata
        self._last_risk_error = error

    def _risk_context_available(self) -> bool:
        return bool(self._last_risk_metadata) or bool(self._last_risk_error)

    def describe_risk_context(self) -> dict[str, object]:
        """Expose the most recently captured deterministic risk context snapshot."""

        return describe_risk_context(self._last_risk_metadata, self._last_risk_error)

    async def stop(self) -> None:
        """Stop the broker interface."""
        if not self.running:
            return

        self.running = False
        task = self._trade_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._trade_task = None
        logger.info("fix_broker_stopped")

    def _resolve_portfolio_state(self, symbol: str) -> Mapping[str, Any]:
        provider = self._portfolio_state_provider
        if provider is None:
            return {}
        try:
            state = provider(symbol)
        except Exception:
            logger.debug("portfolio_state_provider_failed", exc_info=True)
            return {}
        if isinstance(state, Mapping):
            return state
        if isinstance(state, MutableMapping):
            return state
        return {}

    @staticmethod
    def _maybe_get(intent: Any, *names: str) -> Any:
        if isinstance(intent, Mapping):
            for name in names:
                if name in intent:
                    return intent[name]
        for name in names:
            if hasattr(intent, name):
                return getattr(intent, name)
        return None

    @classmethod
    def _extract_quantity(cls, intent: Any, default: float) -> float:
        value = cls._maybe_get(intent, "quantity", "size", "volume")
        if value is None:
            return float(default)
        try:
            if isinstance(value, Decimal):
                return float(value)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _extract_side(cls, intent: Any, fallback: str) -> str:
        value = cls._maybe_get(intent, "side", "direction")
        if isinstance(value, str) and value:
            return value.upper()
        return fallback.upper()

    @staticmethod
    def _resolve_metadata(intent: Any) -> dict[str, Any] | None:
        candidate = None
        if isinstance(intent, Mapping):
            candidate = intent.get("metadata")
        elif hasattr(intent, "metadata"):
            candidate = getattr(intent, "metadata")
        if isinstance(candidate, MutableMapping):
            return dict(candidate)
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return None
        return resolved

    def _build_manual_intent(
        self, symbol: str, side: str, quantity: float, portfolio_state: Mapping[str, Any]
    ):
        metadata: dict[str, Any] = {
            "source": "fix_manual_order",
        }
        current_price = self._coerce_float(portfolio_state.get("current_price"))
        if current_price is not None and current_price > 0:
            metadata.setdefault("reference_price", current_price)

        intent = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": Decimal(str(quantity)),
            "confidence": 1.0,
            "metadata": metadata,
        }
        if current_price is not None and current_price > 0:
            intent["price"] = current_price
        return intent

    async def _publish_risk_rejection(
        self,
        symbol: str,
        side: str,
        quantity: float,
        decision: Mapping[str, Any] | None,
    ) -> None:
        if not self.event_bus or not hasattr(self.event_bus, "emit"):
            return

        self._capture_risk_context()

        reason = "risk_rejected"
        if isinstance(decision, Mapping):
            reason = str(decision.get("reason") or reason)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": float(quantity),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fix_broker_interface",
        }
        payload["runbook"] = _MANUAL_FIX_RISK_RUNBOOK
        if decision:
            payload["decision"] = dict(decision)

        snapshot_payload: dict[str, object] | None = None
        gateway = self._risk_gateway
        if gateway is not None:
            try:
                snapshot = gateway.get_last_policy_snapshot()
            except Exception:
                logger.debug("risk_gateway_policy_snapshot_failed", exc_info=True)
            else:
                if snapshot is not None:
                    try:
                        snapshot_payload = (
                            snapshot.as_dict()
                            if hasattr(snapshot, "as_dict")
                            else dict(snapshot)
                        )
                    except Exception:
                        logger.debug("policy_snapshot_serialisation_failed", exc_info=True)
        if snapshot_payload:
            payload["policy_snapshot"] = snapshot_payload

        policy_violation = True
        violations: list[str] = []
        if snapshot_payload:
            approved = bool(snapshot_payload.get("approved", False))
            raw_violations = snapshot_payload.get("violations")
            if isinstance(raw_violations, (list, tuple)):
                violations = [str(item) for item in raw_violations if item]
            policy_violation = (not approved) or bool(violations)
        if violations:
            payload["violations"] = violations
        payload["policy_violation"] = policy_violation
        payload["severity"] = "critical" if policy_violation else "warning"

        reference_candidates: list[Mapping[str, object]] = []
        if gateway is not None:
            try:
                limits_payload = gateway.get_risk_limits()
            except Exception:
                logger.debug("risk_gateway_limits_failed", exc_info=True)
            else:
                if isinstance(limits_payload, Mapping):
                    candidate_reference: dict[str, object] = {}
                    limits_section = limits_payload.get("limits")
                    if isinstance(limits_section, Mapping):
                        candidate_reference["limits"] = dict(limits_section)
                    summary = limits_payload.get("risk_config_summary")
                    if isinstance(summary, Mapping):
                        candidate_reference["risk_config_summary"] = dict(summary)
                    runbook = limits_payload.get("runbook")
                    if isinstance(runbook, str) and runbook:
                        candidate_reference["risk_api_runbook"] = runbook
                    if candidate_reference:
                        reference_candidates.append(candidate_reference)
        provider = self._risk_interface_provider
        if provider is not None:
            try:
                candidate = provider()
            except Exception:
                logger.debug("risk_interface_provider_failed", exc_info=True)
            else:
                reference = self._normalise_risk_interface(candidate)
                if reference:
                    reference_candidates.append(reference)
        if reference_candidates:
            payload["risk_reference"] = merge_risk_references(
                *reference_candidates,
                runbook=RISK_API_RUNBOOK,
            )

        if self._risk_context_available():
            payload["risk_context"] = self.describe_risk_context()

        try:
            await self.event_bus.emit(self._risk_event_topic, payload)
        except Exception as exc:
            logger.debug("risk_violation_emit_failed", error=str(exc))

    @staticmethod
    def _normalise_risk_interface(candidate: Any) -> dict[str, object]:
        """Convert ``candidate`` into a serialisable risk reference payload."""

        reference: dict[str, object] = {}
        if isinstance(candidate, TradingRiskInterface):
            reference["risk_config_summary"] = candidate.summary()
            if candidate.status is not None:
                reference["risk_interface_status"] = dict(candidate.status)
            reference["risk_config"] = candidate.config.dict()
            return reference

        if isinstance(candidate, Mapping):
            summary = candidate.get("summary")
            if isinstance(summary, Mapping):
                reference["risk_config_summary"] = dict(summary)
            status = candidate.get("status")
            if isinstance(status, Mapping):
                reference["risk_interface_status"] = dict(status)
            config = candidate.get("config")
            if isinstance(config, Mapping):
                reference["risk_config"] = dict(config)
            runbook = candidate.get("risk_api_runbook") or candidate.get("runbook")
            if isinstance(runbook, str) and runbook:
                reference["risk_api_runbook"] = runbook
        return reference

    async def _process_trade_messages(self) -> None:
        """Process trade messages from the queue."""
        try:
            while True:
                try:
                    message = await self.trade_queue.get()
                except asyncio.CancelledError:
                    break

                if not self.running:
                    if message is None:
                        break
                    continue

                if message is None:
                    continue

                # Process based on message type
                msg_type = message.get(35)

                if msg_type == b"8":  # Execution Report
                    await self._handle_execution_report(message)
                elif msg_type == b"9":  # Order Cancel Reject
                    await self._handle_order_cancel_reject(message)

        except asyncio.CancelledError:
            logger.debug("fix_broker_trade_task_cancelled")
        except Exception as e:
            logger.error("trade_message_processing_error", error=str(e))

    def _spawn_task(
        self, coro: Coroutine[Any, Any, Any], *, name: str | None = None
    ) -> asyncio.Task[Any]:
        if self._task_factory is not None:
            return self._task_factory(coro, name)
        return asyncio.create_task(coro, name=name)

    async def _handle_execution_report(self, message: Any) -> None:
        """Handle execution report messages from FIX.

        Expected tags (best-effort):
          - 11 ClOrdID (bytes)
          - 150 ExecType (bytes) [0=New, 1=PartialFill, 2=Fill, 4=Cancelled, 8=Rejected]
          - 32 LastQty (bytes/str)
          - 31 LastPx (bytes/str)
        """
        try:
            order_id = message.get(11).decode() if message.get(11) else None
            exec_type = message.get(150).decode() if message.get(150) else None
            last_qty_raw = message.get(32)
            last_px_raw = message.get(31)
            cum_qty_raw = message.get(14)
            leaves_qty_raw = message.get(151)

            def _as_float(value: Any) -> float | None:
                if value is None:
                    return None
                try:
                    if isinstance(value, (bytes, bytearray)):
                        value = value.decode()
                    return float(value)
                except (TypeError, ValueError):
                    return None

            last_qty = _as_float(last_qty_raw)
            last_px = _as_float(last_px_raw)
            cum_qty = _as_float(cum_qty_raw)
            leaves_qty = _as_float(leaves_qty_raw)

            if not order_id or not exec_type:
                return

            with order_logging_context(order_id, exec_type=exec_type) as order_log:
                order_log.info("execution_report_received")

            # Update in-memory order state
            order_state = self.orders.get(
                order_id,
                {
                    "symbol": None,
                    "side": None,
                    "quantity": 0.0,
                    "status": "UNKNOWN",
                    "timestamp": datetime.utcnow(),
                    "filled_qty": 0.0,
                    "avg_px": None,
                },
            )

            # Map ExecType to status string
            status_map = {
                "0": "ACKNOWLEDGED",
                "1": "PARTIALLY_FILLED",
                "2": "FILLED",
                "4": "CANCELLED",
                "8": "REJECTED",
            }
            new_status = status_map.get(exec_type, order_state.get("status", "UNKNOWN"))
            order_state["status"] = new_status

            if last_qty is not None and last_qty > 0:
                # Update running filled quantity and average price
                filled_val = order_state.get("filled_qty", 0.0)
                prev_filled: float = (
                    float(filled_val) if isinstance(filled_val, (int, float)) else 0.0
                )
                prev_avg_obj = order_state.get("avg_px")
                prev_avg: Optional[float] = (
                    float(prev_avg_obj) if isinstance(prev_avg_obj, (int, float)) else None
                )
                new_filled = float(prev_filled) + float(last_qty)
                if cum_qty is not None and cum_qty >= 0:
                    new_filled = max(new_filled, float(cum_qty))
                if last_px is not None:
                    if prev_avg is None or prev_filled <= 0.0:
                        new_avg: float = float(last_px)
                    else:
                        total_value: float = float(prev_avg) * float(prev_filled) + float(
                            last_px
                        ) * float(last_qty)
                        new_avg = (
                            float(total_value / new_filled) if new_filled > 0.0 else float(prev_avg)
                        )
                    order_state["avg_px"] = new_avg
                order_state["filled_qty"] = new_filled
            if cum_qty is not None:
                order_state["filled_qty"] = cum_qty
            if leaves_qty is not None:
                order_state["leaves_qty"] = leaves_qty

            self.orders[order_id] = order_state

            update_payload = {
                "order_id": order_id,
                "exec_type": exec_type,
                "status": order_state.get("status"),
                "filled_qty": order_state.get("filled_qty"),
                "avg_px": order_state.get("avg_px"),
                "symbol": order_state.get("symbol"),
                "side": order_state.get("side"),
                "timestamp": datetime.utcnow(),
            }

            update_payload["last_qty"] = last_qty
            update_payload["last_px"] = last_px
            if cum_qty is not None:
                update_payload["cum_qty"] = cum_qty
            if leaves_qty is not None:
                update_payload["leaves_qty"] = leaves_qty

            # Emit event for system (if compatible bus provided)
            try:
                if self.event_bus and hasattr(self.event_bus, "emit"):
                    await self.event_bus.emit("order_update", update_payload)
            except Exception as emit_err:
                with order_logging_context(order_id, exec_type=exec_type):
                    logger.debug("event_bus_emit_failed", error=str(emit_err))

            # Notify local listeners
            await self._notify_listeners("order_update", order_id, update_payload)

            exec_event_map = {
                "0": "acknowledged",
                "1": "partial_fill",
                "2": "filled",
                "4": "cancelled",
                "8": "rejected",
            }
            event_type = exec_event_map.get(exec_type)
            if event_type:
                await self._notify_listeners(event_type, order_id, update_payload)

        except Exception as e:
            with order_logging_context(order_id or "unknown"):
                logger.error("execution_report_error", error=str(e))

    async def _handle_order_cancel_reject(self, message: Any) -> None:
        """Handle order cancel reject messages."""
        try:
            order_id = message.get(11).decode() if message.get(11) else None
            reject_reason = message.get(58).decode() if message.get(58) else "Unknown"

            if order_id:
                with order_logging_context(order_id, correlation_id=order_id):
                    logger.warning(
                        "order_cancel_rejected", reason=reject_reason
                    )

                # Emit event for system
                payload = {
                    "order_id": order_id,
                    "reason": reject_reason,
                    "timestamp": datetime.utcnow(),
                }

                if self.event_bus and hasattr(self.event_bus, "emit"):
                    await self.event_bus.emit(
                        "order_cancel_rejected",
                        payload,
                    )

                await self._notify_listeners("cancel_rejected", order_id, payload)

        except Exception as e:
            with order_logging_context(order_id or "unknown"):
                logger.error("order_cancel_reject_error", error=str(e))

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            self._capture_risk_context()
            risk_gateway = self._risk_gateway
            portfolio_state: Mapping[str, Any] = {}
            if risk_gateway is not None:
                portfolio_state = self._resolve_portfolio_state(symbol)
                intent = self._build_manual_intent(symbol, side, quantity, portfolio_state)
                validated_intent = await risk_gateway.validate_trade_intent(
                    intent, portfolio_state
                )
                if not validated_intent:
                    decision: Mapping[str, Any] | None = None
                    try:
                        decision = risk_gateway.get_last_decision()
                    except Exception:
                        logger.debug("risk_gateway_last_decision_failed", exc_info=True)
                    await self._publish_risk_rejection(symbol, side, quantity, decision)
                    logger.warning(
                        "order_blocked_by_risk_gateway",
                        symbol=symbol,
                        side=side,
                    )
                    return None

                quantity = self._extract_quantity(validated_intent, quantity)
                side = self._extract_side(validated_intent, side)
                metadata = self._resolve_metadata(validated_intent)
            else:
                metadata = None
            # Generate order ID
            order_id = f"ORD_{int(datetime.utcnow().timestamp() * 1000)}"

            # Create order message
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")  # NewOrderSingle
            msg.append_pair(11, order_id)  # ClOrdID
            msg.append_pair(55, symbol)  # Symbol
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(
                60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
            )  # TransactTime

            # Send order
            if self.fix_initiator:
                self.fix_initiator.send_message(msg)
                with order_logging_context(order_id) as order_log:
                    order_log.info(
                        "market_order_submitted",
                        symbol=symbol,
                        side=side.upper(),
                        quantity=float(quantity),
                    )

                # Store order
                self.orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": "PENDING",
                    "timestamp": datetime.utcnow(),
                }
                if metadata:
                    self.orders[order_id]["metadata"] = metadata
                if risk_gateway is not None:
                    decision: Mapping[str, Any] | None = None
                    try:
                        decision = risk_gateway.get_last_decision()
                    except Exception:
                        logger.debug("risk_gateway_last_decision_failed", exc_info=True)
                    if decision:
                        self.orders[order_id]["risk_decision"] = dict(decision)
                    try:
                        snapshot = risk_gateway.get_last_policy_snapshot()
                    except Exception:
                        logger.debug("risk_gateway_policy_snapshot_failed", exc_info=True)
                    else:
                        if snapshot is not None:
                            try:
                                self.orders[order_id]["policy_snapshot"] = (
                                    snapshot.as_dict()
                                    if hasattr(snapshot, "as_dict")
                                    else dict(snapshot)
                                )
                            except Exception:
                                logger.debug(
                                    "policy_snapshot_serialisation_failed", exc_info=True
                                )
                if self._risk_context_available():
                    self.orders[order_id]["risk_context"] = self.describe_risk_context()

                return order_id
        except Exception as e:
            with order_logging_context(locals().get("order_id", "unknown")):
                logger.error("market_order_error", error=str(e))
            return None
        # If not sent (e.g., no fix_initiator), return None
        return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancel request sent, False otherwise
        """
        try:
            # Minimal cTrader schema: 11/41 only
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "F")  # OrderCancelRequest
            cncl_id = f"CNCL_{order_id}"
            msg.append_pair(11, cncl_id)
            msg.append_pair(41, order_id)

            if self.fix_initiator:
                self.fix_initiator.send_message(msg)
                with order_logging_context(order_id):
                    logger.info("order_cancel_requested")
                return True

        except Exception as e:
            with order_logging_context(order_id):
                logger.error("order_cancel_error", error=str(e))
        # Ensure a bool is always returned
        return False

    def get_order_status(self, order_id: str) -> Optional[dict[str, object]]:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order status dictionary or None
        """
        return self.orders.get(order_id)

    def get_all_orders(self) -> dict[str, dict[str, object]]:
        """Get all orders."""
        return self.orders.copy()

    # --- Listener registration -------------------------------------------------
    def add_order_update_listener(
        self,
        callback: OrderEventCallback,
    ) -> bool:
        """Register a callback to receive order update notifications.

        Callback signature: (order_id: str, update: dict[str, Any]) -> None | Awaitable
        """
        try:
            self._order_update_listeners.append(callback)
            self._event_callbacks["order_update"].append(callback)
            return True
        except Exception:
            return False

    def remove_order_update_listener(
        self,
        callback: OrderEventCallback,
    ) -> bool:
        """Unregister a previously added callback."""
        try:
            if callback in self._order_update_listeners:
                self._order_update_listeners.remove(callback)
            if callback in self._event_callbacks.get("order_update", []):
                self._event_callbacks["order_update"].remove(callback)
                return True
            return False
        except Exception:
            return False

    # --- New event registration API -------------------------------------
    def add_event_listener(self, event_type: str, callback: OrderEventCallback) -> bool:
        """Register a callback for a specific order lifecycle event."""

        if event_type not in ORDER_EVENT_TYPES:
            raise ValueError(f"Unsupported order event type: {event_type}")
        try:
            self._event_callbacks[event_type].append(callback)
            return True
        except Exception:
            return False

    def remove_event_listener(self, event_type: str, callback: OrderEventCallback) -> bool:
        """Remove a previously registered event listener."""

        if event_type not in ORDER_EVENT_TYPES:
            return False
        try:
            callbacks = self._event_callbacks.get(event_type, [])
            if callback in callbacks:
                callbacks.remove(callback)
                return True
            return False
        except Exception:
            return False

    async def _notify_listeners(
        self,
        event_type: str,
        order_id: str,
        payload: dict[str, Any],
    ) -> None:
        callbacks = list(self._event_callbacks.get(event_type, ()))
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order_id, payload)
                else:
                    callback(order_id, payload)
            except Exception as cb_err:
                with order_logging_context(order_id, event_type=event_type):
                    logger.warning(
                        "order_event_listener_error",
                        error=str(cb_err),
                    )
