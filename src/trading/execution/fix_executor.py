"""Lightweight FIX execution shim backed by roadmap lifecycle primitives.

The original FIX executor in the proving ground acted as a thin stub that
mutated the :class:`~src.trading.models.order.Order` directly and tracked
positions with ad-hoc bookkeeping.  The roadmap work introduced a proper order
lifecycle state machine and an inventory-aware position tracker that handles
FIFO/LIFO accounting, realised PnL, and exposure snapshots.  This module now
leans on those primitives so higher level components exercising the legacy
``FIXExecutor`` gain the same determinism and telemetry as the FIX broker
interface without rewriting their call sites yet.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

try:
    from src.core.interfaces import IExecutionEngine
except Exception:  # pragma: no cover

    class IExecutionEngine:  # type: ignore
        pass


if TYPE_CHECKING:
    from src.core.interfaces import IExecutionEngine as _IExecForTyping  # noqa: F401
else:

    class _IExecForTyping:  # runtime-friendly base if needed
        pass


try:
    from src.trading.models.order import Order, OrderStatus
    from src.trading.models.position import Position
    from src.trading.order_management import (
        OrderEventType,
        OrderLifecycle,
        OrderLifecycleSnapshot,
        OrderStateError,
        PositionSnapshot,
        PositionTracker,
    )
except Exception:  # pragma: no cover

    class Order:  # type: ignore[no-redef]
        pass

    class Position:  # type: ignore[no-redef]
        pass

    class OrderStatus:  # type: ignore
        PENDING = type("EnumVal", (), {"value": "PENDING"})()
        FILLED = type("EnumVal", (), {"value": "FILLED"})()
        CANCELLED = type("EnumVal", (), {"value": "CANCELLED"})()

    class OrderLifecycle:  # type: ignore[no-redef]
        def __init__(self, order_id: str, **_: object) -> None:
            self.order_id = order_id

        def snapshot(self) -> "OrderLifecycleSnapshot":  # pragma: no cover - fallback
            return OrderLifecycleSnapshot(order_id=self.order_id, status=OrderStatus.PENDING)  # type: ignore[arg-type]

        def apply_event(self, *_: object, **__: object) -> "OrderLifecycleSnapshot":
            return self.snapshot()

        def apply_fix_execution(self, *_: object, **__: object) -> "OrderLifecycleSnapshot":
            return self.snapshot()

    class OrderLifecycleSnapshot:  # type: ignore[no-redef]
        def __init__(self, order_id: str, status: object) -> None:
            self.order_id = order_id
            self.status = status
            self.filled_quantity = 0.0
            self.remaining_quantity = 0.0
            self.average_price = None

    class PositionTracker:  # type: ignore[no-redef]
        def __init__(self, **_: object) -> None:
            self._positions: Dict[str, PositionSnapshot] = {}

        def record_fill(self, symbol: str, quantity: float, price: float, **_: object) -> PositionSnapshot:
            snapshot = self._positions.get(symbol)
            if snapshot is None:
                snapshot = PositionSnapshot(symbol, quantity, quantity, 0.0, price, price, price, 0.0, 0.0, quantity * price)  # type: ignore[arg-type]
            self._positions[symbol] = snapshot
            return snapshot

        def get_position_snapshot(self, symbol: str, **_: object) -> "PositionSnapshot":
            return self._positions.setdefault(symbol, PositionSnapshot(symbol, "ACC", 0.0, 0.0, 0.0, None, None, None, 0.0, None, None))  # type: ignore[arg-type]

    class PositionSnapshot:  # type: ignore[no-redef]
        def __init__(
            self,
            symbol: str,
            account: str,
            net_quantity: float,
            long_quantity: float,
            short_quantity: float,
            market_price: Optional[float],
            average_long_price: Optional[float],
            average_short_price: Optional[float],
            realized_pnl: float,
            unrealized_pnl: Optional[float],
            exposure: Optional[float],
        ) -> None:
            self.symbol = symbol
            self.account = account
            self.net_quantity = net_quantity
            self.long_quantity = long_quantity
            self.short_quantity = short_quantity
            self.market_price = market_price
            self.average_long_price = average_long_price
            self.average_short_price = average_short_price
            self.realized_pnl = realized_pnl
            self.unrealized_pnl = unrealized_pnl
            self.exposure = exposure


logger = logging.getLogger(__name__)


class FIXExecutor(IExecutionEngine):
    """
    FIX protocol-based execution engine for managing order execution
    and position management through FIX API connections.
    """

    def __init__(self, fix_config: Optional[dict[str, object]] = None):
        self.fix_config = fix_config or {}
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.execution_history: List[dict[str, object]] = []
        self.is_initialized = False
        self._lifecycles: Dict[str, OrderLifecycle] = {}
        self._position_tracker = PositionTracker()

    async def initialize(self) -> bool:
        """Initialize the FIX executor."""
        try:
            logger.info("Initializing FIX executor (deprecated stub)...")

            # Initialize FIX connection
            # In real implementation, this would establish FIX session
            self.is_initialized = True

            logger.info("FIX executor initialized successfully (deprecated)")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FIX executor: {e}")
            return False

    async def execute_order(self, order: Order) -> bool:
        """
        Execute a trading order via FIX protocol.

        Args:
            order: The order to execute

        Returns:
            True if execution successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("FIX executor not initialized")
                return False

            logger.info(
                f"Executing FIX order: {order.order_id} - {order.symbol} {order.side} {order.quantity}"
            )

            # Validate order
            if not self._validate_order(order):
                return False

            # Add to active orders while we simulate the FIX round-trip
            self.active_orders[order.order_id] = order
            order.status = OrderStatus.PENDING

            # In real implementation, send order via FIX.  The proving ground still
            # simulates fills locally, so we exercise the lifecycle state machine
            # and inventory tracker directly.
            fill_snapshot, fill_quantity, fill_price = await self._simulate_fix_execution(order)

            # Update position and realised PnL via the roadmap tracker
            position_snapshot = self._update_position(order, fill_quantity, fill_price)

            # Log execution
            self._log_execution(order, fill_snapshot, position_snapshot)

            # Filled orders no longer count as active
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                self.active_orders.pop(order.order_id, None)

            logger.info(f"FIX order executed successfully: {order.order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to execute FIX order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order via FIX protocol.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("FIX executor not initialized")
                return False

            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False

            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]

            lifecycle = self._lifecycles.get(order_id)
            if lifecycle is not None:
                try:
                    lifecycle.apply_event(
                        OrderEventType.CANCELLED,
                        timestamp=datetime.now(timezone.utc),
                        reason="ClientCancelRequest",
                        leaves_quantity=max(order.quantity - order.filled_quantity, 0.0),
                    )
                except OrderStateError:
                    logger.debug("Lifecycle already terminal for cancelled order %s", order_id)

            # In real implementation, send cancel via FIX
            logger.info(f"FIX order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel FIX order {order_id}: {e}")
            return False

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current position or None if no position
        """
        return self.positions.get(symbol)

    async def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.

        Returns:
            List of active orders
        """
        return list(self.active_orders.values())

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if not order.symbol:
            logger.error("Order symbol is required")
            return False

        if order.quantity <= 0:
            logger.error("Order quantity must be positive")
            return False

        allowed_types: set[str] = {"MARKET", "LIMIT", "STOP"}
        otype = getattr(order, "order_type", None)
        if isinstance(otype, Enum):
            normalized_type = str(otype.value).upper()
        else:
            normalized_type = str(otype or "").upper()
        if normalized_type not in allowed_types:
            logger.error(f"Invalid order type: {order.order_type}")
            return False

        return True

    async def _simulate_fix_execution(
        self, order: Order
    ) -> tuple[OrderLifecycleSnapshot, float, float]:
        """Simulate FIX order execution using the roadmap lifecycle."""

        await asyncio.sleep(0.05)

        lifecycle = self._lifecycles.get(order.order_id)
        if lifecycle is None:
            lifecycle = OrderLifecycle(
                order.order_id,
                quantity=float(order.quantity),
                symbol=order.symbol,
                side=order.side,
                created_at=datetime.now(timezone.utc),
            )
            self._lifecycles[order.order_id] = lifecycle
        else:
            lifecycle.set_initial_quantity(float(order.quantity))

        now = datetime.now(timezone.utc)
        try:
            lifecycle.apply_event(OrderEventType.ACKNOWLEDGED, timestamp=now)
        except OrderStateError:
            logger.debug("Order %s already acknowledged", order.order_id)

        fill_price_source = order.price if order.price is not None else order.average_price
        fill_price = float(fill_price_source or 0.0)
        existing_filled = float(order.filled_quantity)
        fill_quantity = max(float(order.quantity) - existing_filled, 0.0)
        leaves = max(float(order.quantity) - (existing_filled + fill_quantity), 0.0)
        snapshot = lifecycle.apply_event(
            OrderEventType.FILL,
            timestamp=now,
            quantity=fill_quantity,
            price=fill_price,
            leaves_quantity=max(leaves, 0.0),
        )

        new_fill = snapshot.filled_quantity - order.filled_quantity
        if new_fill > 0:
            order.add_fill(new_fill, fill_price)

        return snapshot, new_fill, fill_price

    def _update_position(
        self, order: Order, fill_quantity: float, fill_price: float
    ) -> PositionSnapshot:
        """Update position based on executed order via :class:`PositionTracker`."""

        if fill_quantity <= 0:
            return self._position_tracker.get_position_snapshot(order.symbol)

        if order.average_price is None:
            order.average_price = fill_price

        signed_quantity = float(fill_quantity)
        if order.side.upper() == "SELL":
            signed_quantity = -signed_quantity

        snapshot = self._position_tracker.record_fill(
            order.symbol,
            signed_quantity,
            float(fill_price if order.price is None else order.average_price or fill_price),
            timestamp=datetime.now(timezone.utc),
        )

        # Keep the latest execution price available as a mark for exposure metrics.
        self._position_tracker.update_mark_price(
            order.symbol,
            fill_price,
            timestamp=datetime.now(timezone.utc),
        )

        position = self.positions.get(order.symbol)
        if position is None:
            position = Position(
                symbol=order.symbol,
                quantity=snapshot.net_quantity,
                average_price=(snapshot.average_long_price or snapshot.average_short_price or float(order.average_price)),
                realized_pnl=snapshot.realized_pnl,
                market_price=snapshot.market_price,
            )
            self.positions[order.symbol] = position
        else:
            if snapshot.net_quantity == 0:
                position.update_quantity(0.0, 0.0)
            else:
                avg_price = snapshot.average_long_price if snapshot.net_quantity > 0 else snapshot.average_short_price
                position.update_quantity(snapshot.net_quantity, avg_price or float(order.average_price))
            position.realized_pnl = snapshot.realized_pnl
            if snapshot.market_price is not None:
                position.update_market_price(snapshot.market_price)

        return snapshot

    def _log_execution(
        self,
        order: Order,
        lifecycle_snapshot: OrderLifecycleSnapshot,
        position_snapshot: PositionSnapshot,
    ) -> None:
        """Log order execution details enriched with lifecycle telemetry."""

        execution_record = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "average_price": order.average_price,
            "status": order.status.value,
            "timestamp": order.filled_at.isoformat() if order.filled_at else None,
            "lifecycle_status": lifecycle_snapshot.status.value
            if hasattr(lifecycle_snapshot.status, "value")
            else lifecycle_snapshot.status,
            "remaining_quantity": lifecycle_snapshot.remaining_quantity,
            "position_net_quantity": position_snapshot.net_quantity,
            "position_realized_pnl": position_snapshot.realized_pnl,
        }
        self.execution_history.append(execution_record)

    def get_order_snapshot(self, order_id: str) -> Optional[OrderLifecycleSnapshot]:
        """Return the latest lifecycle snapshot for ``order_id`` if tracked."""

        lifecycle = self._lifecycles.get(order_id)
        if lifecycle is None:
            return None
        return lifecycle.snapshot()
