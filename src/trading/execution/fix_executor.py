"""
DEPRECATED: FIXExecutor
=======================

This module is superseded by the consolidated `FIXBrokerInterface` and
high-level order lifecycle/position tracking. It remains as a stub for
backward compatibility and will be removed after migration.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

from ._risk_context import (
    RiskContextProvider,
    capture_risk_context,
    describe_risk_context,
)

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
except Exception:  # pragma: no cover

    class Order:  # type: ignore[no-redef]
        pass

    class Position:  # type: ignore[no-redef]
        pass

    class OrderStatus:  # type: ignore
        PENDING = type("EnumVal", (), {"value": "PENDING"})()
        FILLED = type("EnumVal", (), {"value": "FILLED"})()
        CANCELLED = type("EnumVal", (), {"value": "CANCELLED"})()


logger = logging.getLogger(__name__)


class FIXExecutor(IExecutionEngine):
    """
    FIX protocol-based execution engine for managing order execution
    and position management through FIX API connections.
    """

    def __init__(
        self,
        fix_config: Optional[dict[str, object]] = None,
        *,
        risk_context_provider: RiskContextProvider | None = None,
    ):
        self.fix_config = fix_config or {}
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.execution_history: List[dict[str, object]] = []
        self.is_initialized = False
        self._risk_context_provider: RiskContextProvider | None = None
        self._last_risk_metadata: dict[str, object] | None = None
        self._last_risk_error: dict[str, object] | None = None
        if risk_context_provider is not None:
            self.set_risk_context_provider(risk_context_provider)

    def set_risk_context_provider(self, provider: RiskContextProvider | None) -> None:
        """Install or replace the callable that resolves trading risk metadata."""

        if provider is not None and not callable(provider):
            raise TypeError("risk_context_provider must be callable or None")
        self._risk_context_provider = provider

    def _capture_risk_context(self) -> None:
        metadata, error = capture_risk_context(self._risk_context_provider)
        self._last_risk_metadata = metadata
        self._last_risk_error = error

    def describe_risk_context(self) -> dict[str, object]:
        """Expose the most recent deterministic risk context snapshot."""

        return describe_risk_context(self._last_risk_metadata, self._last_risk_error)

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
            self._capture_risk_context()

            if not self.is_initialized:
                logger.error("FIX executor not initialized")
                return False

            logger.info(
                f"Executing FIX order: {order.order_id} - {order.symbol} {order.side} {order.quantity}"
            )

            # Validate order
            if not self._validate_order(order):
                return False

            # Add to active orders
            self.active_orders[order.order_id] = order
            order.status = OrderStatus.PENDING

            # In real implementation, send order via FIX
            # For now, simulate successful execution
            await self._simulate_fix_execution(order)

            # Update position
            await self._update_position(order)

            # Log execution
            self._log_execution(order)

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

    async def _simulate_fix_execution(self, order: Order) -> None:
        """Simulate FIX order execution."""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simulate successful execution using order data
        fill_price_source = order.price
        if fill_price_source is None:
            fill_price_source = order.average_price
        fill_price = float(fill_price_source or 0.0)
        order.add_fill(order.quantity, fill_price)

    async def _update_position(self, order: Order) -> None:
        """Update position based on executed order."""
        symbol = order.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=0.0, average_price=0.0, unrealized_pnl=0.0, realized_pnl=0.0
            )

        position = self.positions[symbol]

        if order.side == "BUY":
            current_quantity = position.quantity
            current_value = current_quantity * position.average_price
            new_quantity = current_quantity + order.quantity
            new_value = current_value + order.quantity * order.average_price
            position.quantity = new_quantity
            position.average_price = new_value / new_quantity if new_quantity != 0 else 0.0
        elif order.side == "SELL":
            position.quantity -= order.quantity
            realized = (order.average_price - position.average_price) * order.quantity
            position.realized_pnl += realized
            if position.quantity <= 0:
                position.quantity = 0.0
                position.average_price = 0.0

    def _log_execution(self, order: Order) -> None:
        """Record execution details for audit trail."""
        entry = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.average_price,
            "status": order.status.value,
        }
        self.execution_history.append(entry)


__all__ = ["FIXExecutor"]
