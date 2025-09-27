"""
FIX Broker Interface for IC Markets
Provides integration between FIX protocol and trading system
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone

# Add typing for callbacks and awaitables
from typing import Any, Awaitable, Callable, Coroutine, Optional

import simplefix

from ..order_management.order_state_machine import (
    OrderEventType,
    OrderLifecycle,
)

logger = logging.getLogger(__name__)


TaskFactory = Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _decode_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:  # pragma: no cover - defensive
            return None
    return str(value)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return float(value.decode())
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive
        return None


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
        self.orders: dict[str, OrderLifecycle] = {}
        self._order_update_listeners: list[
            Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]
        ] = []  # callbacks taking (order_id: str, update: dict[str, Any])
        self._trade_task: asyncio.Task[Any] | None = None
        self._task_factory = task_factory

    async def start(self) -> None:
        """Start the broker interface."""
        if self.running:
            return

        self.running = True
        logger.info("FIX broker interface started")

        # Start message processing
        self._trade_task = self._spawn_task(
            self._process_trade_messages(),
            name="fix-broker-trade-feed",
        )

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
        logger.info("FIX broker interface stopped")

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
            logger.debug("FIX broker trade task cancelled")
        except Exception as e:
            logger.error(f"Error processing trade message: {e}")

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
            last_qty = None
            last_px = None
            try:
                if last_qty_raw is not None:
                    last_qty = float(
                        last_qty_raw.decode()
                        if isinstance(last_qty_raw, (bytes, bytearray))
                        else last_qty_raw
                    )
                if last_px_raw is not None:
                    last_px = float(
                        last_px_raw.decode()
                        if isinstance(last_px_raw, (bytes, bytearray))
                        else last_px_raw
                    )
            except Exception:
                # Non-fatal; continue without qty/px
                pass

            if not order_id or not exec_type:
                return

            logger.info(f"Execution report for order {order_id}: {exec_type}")

            symbol = _decode_str(message.get(55))
            side_tag = _decode_str(message.get(54))
            side = None
            if side_tag == "1":
                side = "BUY"
            elif side_tag == "2":
                side = "SELL"

            lifecycle = self.orders.get(order_id)
            if lifecycle is None:
                lifecycle = OrderLifecycle(
                    order_id,
                    quantity=_coerce_float(message.get(38)),
                    symbol=symbol,
                    side=side,
                    created_at=_utc_now(),
                )
                self.orders[order_id] = lifecycle
            else:
                if lifecycle.symbol is None and symbol is not None:
                    lifecycle.symbol = symbol
                if lifecycle.side is None and side is not None:
                    lifecycle.side = side
                if (
                    lifecycle.initial_quantity is None
                    and (order_qty := _coerce_float(message.get(38))) is not None
                ):
                    lifecycle.set_initial_quantity(order_qty)

            leaves_qty_raw = message.get(151)
            reason = _decode_str(message.get(58)) or _decode_str(message.get(103))

            try:
                snapshot = lifecycle.apply_fix_execution(
                    exec_type,
                    last_qty=last_qty,
                    last_px=last_px,
                    leaves_qty=leaves_qty_raw,
                    reason=reason,
                    timestamp=_utc_now(),
                )
            except Exception as exc:
                logger.error(f"Error applying execution report for {order_id}: {exc}")
                return

            update_payload = {
                **snapshot.to_dict(),
                "exec_type": exec_type,
                "last_qty": last_qty,
                "last_px": last_px,
            }

            # Emit event for system (if compatible bus provided)
            try:
                if self.event_bus and hasattr(self.event_bus, "emit"):
                    await self.event_bus.emit("order_update", update_payload)
            except Exception as emit_err:
                logger.debug(f"Event bus emit failed: {emit_err}")

            # Notify local listeners
            for callback in list(self._order_update_listeners):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order_id, update_payload)
                    else:
                        callback(order_id, update_payload)
                except Exception as cb_err:
                    logger.warning(f"Order update listener error: {cb_err}")

        except Exception as e:
            logger.error(f"Error handling execution report: {e}")

    async def _handle_order_cancel_reject(self, message: Any) -> None:
        """Handle order cancel reject messages."""
        try:
            order_id = _decode_str(message.get(11))
            reject_reason = _decode_str(message.get(58)) or "Unknown"

            if order_id:
                logger.warning(f"Order cancel rejected for {order_id}: {reject_reason}")

                # Emit event for system
                await self.event_bus.emit(
                    "order_cancel_rejected",
                    {"order_id": order_id, "reason": reject_reason, "timestamp": _utc_now()},
                )

        except Exception as e:
            logger.error(f"Error handling order cancel reject: {e}")

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
            # Risk guard: ensure portfolio risk within limits if a risk manager is available on event_bus
            try:
                risk_manager = (
                    getattr(self.event_bus, "risk_manager", None) if self.event_bus else None
                )
                if risk_manager and hasattr(risk_manager, "check_risk_thresholds"):
                    if not risk_manager.check_risk_thresholds():
                        logger.warning("Order blocked by risk thresholds (VaR/ES limits)")
                        return None
            except Exception:
                # If risk check fails, proceed conservatively without blocking
                pass
            # Generate order ID
            order_id = f"ORD_{int(_utc_now().timestamp() * 1000)}"

            # Create order message
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")  # NewOrderSingle
            msg.append_pair(11, order_id)  # ClOrdID
            msg.append_pair(55, symbol)  # Symbol
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(60, _utc_now().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # TransactTime

            # Send order
            if self.fix_initiator:
                self.fix_initiator.send_message(msg)
                logger.info(f"Market order placed: {side} {quantity} {symbol} (ID: {order_id})")

                lifecycle = OrderLifecycle(
                    order_id,
                    quantity=quantity,
                    symbol=symbol,
                    side=side.upper(),
                    created_at=_utc_now(),
                )
                lifecycle.apply_event(OrderEventType.ACKNOWLEDGED, timestamp=_utc_now())
                self.orders[order_id] = lifecycle

                return order_id
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
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
                logger.info(f"Order cancel requested: {order_id}")
                lifecycle = self.orders.get(order_id)
                if lifecycle is not None:
                    try:
                        lifecycle.apply_event(
                            OrderEventType.CANCELLED,
                            timestamp=_utc_now(),
                            reason="ClientCancelRequest",
                        )
                    except Exception:
                        pass
                return True

        except Exception as e:
            logger.error(f"Error canceling order: {e}")
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
        lifecycle = self.orders.get(order_id)
        if lifecycle is None:
            return None
        snapshot = lifecycle.snapshot()
        return snapshot.to_dict()

    def get_all_orders(self) -> dict[str, dict[str, object]]:
        """Get all orders."""
        return {
            order_id: lifecycle.snapshot().to_dict()
            for order_id, lifecycle in self.orders.items()
        }

    # --- Listener registration -------------------------------------------------
    def add_order_update_listener(
        self,
        callback: (
            Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]
        ),
    ) -> bool:
        """Register a callback to receive order update notifications.

        Callback signature: (order_id: str, update: dict[str, Any]) -> None | Awaitable
        """
        try:
            self._order_update_listeners.append(callback)
            return True
        except Exception:
            return False

    def remove_order_update_listener(
        self,
        callback: (
            Callable[[str, dict[str, Any]], None] | Callable[[str, dict[str, Any]], Awaitable[None]]
        ),
    ) -> bool:
        """Unregister a previously added callback."""
        try:
            if callback in self._order_update_listeners:
                self._order_update_listeners.remove(callback)
                return True
            return False
        except Exception:
            return False
