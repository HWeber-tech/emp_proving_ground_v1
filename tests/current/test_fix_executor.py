"""Regression tests for the deprecated FIX executor stub."""

from __future__ import annotations

import pytest

from src.trading.execution.fix_executor import FIXExecutor
from src.trading.models.order import Order, OrderStatus, OrderType


@pytest.mark.asyncio
async def test_execute_order_updates_position_and_history() -> None:
    """Successful execution should fill the order and record bookkeeping."""

    executor = FIXExecutor()
    assert await executor.initialize()

    order = Order(
        order_id="ABC123",
        symbol="EURUSD",
        side="BUY",
        quantity=100_000,
        order_type=OrderType.MARKET,
        price=1.2345,
    )

    assert await executor.execute_order(order)

    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(order.quantity)
    assert order.average_price == pytest.approx(1.2345)

    position = await executor.get_position("EURUSD")
    assert position is not None
    assert position.quantity == pytest.approx(100_000)
    assert position.average_price == pytest.approx(1.2345)

    assert executor.execution_history, "Execution history should record fills"
    history_entry = executor.execution_history[-1]
    assert history_entry["order_id"] == "ABC123"
    assert history_entry["status"] == OrderStatus.FILLED.value
    assert history_entry["lifecycle_status"] == OrderStatus.FILLED.value
    assert history_entry["position_net_quantity"] == pytest.approx(order.quantity)

    snapshot = executor.get_order_snapshot(order.order_id)
    assert snapshot is not None
    assert getattr(snapshot.status, "value", snapshot.status) == OrderStatus.FILLED.value


@pytest.mark.asyncio
async def test_execute_order_requires_initialization() -> None:
    """Orders should not execute until the executor is initialized."""

    executor = FIXExecutor()
    order = Order(
        order_id="NEEDS_INIT",
        symbol="EURUSD",
        side="BUY",
        quantity=1,
        order_type=OrderType.MARKET,
        price=1.1,
    )

    assert not await executor.execute_order(order)
    assert order.status is OrderStatus.PENDING


@pytest.mark.asyncio
async def test_execute_order_rejects_invalid_type() -> None:
    """Invalid order types should be rejected during validation."""

    executor = FIXExecutor()
    assert await executor.initialize()

    order = Order(
        order_id="BADTYPE",
        symbol="EURUSD",
        side="BUY",
        quantity=10,
        order_type=OrderType.MARKET,
        price=1.0,
    )
    order.order_type = "ICECREAM"  # simulate legacy code mutating the enum

    assert not await executor.execute_order(order)
    assert order.order_id not in executor.active_orders


@pytest.mark.asyncio
async def test_execute_order_rejects_non_positive_quantity() -> None:
    """Quantity validation should block zero or negative orders."""

    executor = FIXExecutor()
    assert await executor.initialize()

    zero_order = Order(
        order_id="ZERO",
        symbol="EURUSD",
        side="BUY",
        quantity=0,
        order_type=OrderType.MARKET,
        price=1.0,
    )

    assert not await executor.execute_order(zero_order)
    assert zero_order.order_id not in executor.active_orders

    negative_order = Order(
        order_id="NEG",
        symbol="EURUSD",
        side="SELL",
        quantity=-10,
        order_type=OrderType.MARKET,
        price=1.0,
    )

    assert not await executor.execute_order(negative_order)
    assert negative_order.order_id not in executor.active_orders


@pytest.mark.asyncio
async def test_cancel_order_clears_active_orders() -> None:
    """Active orders should be cancellable once tracked."""

    executor = FIXExecutor()
    assert await executor.initialize()

    order = Order(
        order_id="CAN123",
        symbol="EURUSD",
        side="BUY",
        quantity=5,
        order_type=OrderType.LIMIT,
        price=1.1111,
    )
    executor.active_orders[order.order_id] = order

    assert await executor.cancel_order(order.order_id)
    assert order.status is OrderStatus.CANCELLED
    assert order.order_id not in executor.active_orders


@pytest.mark.asyncio
async def test_cancel_order_missing_returns_false() -> None:
    """Cancelling a non-existent order should fail gracefully."""

    executor = FIXExecutor()
    assert await executor.initialize()

    assert not await executor.cancel_order("UNKNOWN")


@pytest.mark.asyncio
async def test_sell_updates_realized_pnl() -> None:
    """Selling against an open position should accrue realized PnL."""

    executor = FIXExecutor()
    assert await executor.initialize()

    buy_order = Order(
        order_id="BUY1",
        symbol="EURUSD",
        side="BUY",
        quantity=50_000,
        order_type=OrderType.MARKET,
        price=1.0,
    )
    assert await executor.execute_order(buy_order)

    sell_order = Order(
        order_id="SELL1",
        symbol="EURUSD",
        side="SELL",
        quantity=50_000,
        order_type=OrderType.MARKET,
        price=1.1,
    )
    assert await executor.execute_order(sell_order)

    position = await executor.get_position("EURUSD")
    assert position is not None
    assert position.quantity == pytest.approx(0.0)
    expected_pnl = (1.1 - 1.0) * 50_000
    assert position.realized_pnl == pytest.approx(expected_pnl)


@pytest.mark.asyncio
async def test_get_active_orders_returns_copy() -> None:
    """The active orders list should be a defensive copy."""

    executor = FIXExecutor()
    assert await executor.initialize()

    order = Order(
        order_id="COPY1",
        symbol="EURUSD",
        side="BUY",
        quantity=1,
        order_type=OrderType.MARKET,
        price=1.0,
    )

    # Manually register to avoid waiting for execution sleep
    executor.active_orders[order.order_id] = order

    active = await executor.get_active_orders()
    assert len(active) == 1
    active.pop()

    # Original dictionary should remain intact
    assert order.order_id in executor.active_orders
