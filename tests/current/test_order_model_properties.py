"""Property-based regression tests for src.trading.models.order."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.trading.models.order import Order, OrderStatus, OrderType


positive_floats = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
price_floats = st.floats(min_value=1e-5, max_value=1e3, allow_nan=False, allow_infinity=False)


@given(positive_floats, price_floats, price_floats)
def test_add_fill_computes_weighted_average(
    qty: float, first_price: float, second_price: float
) -> None:
    qty_a = max(qty * 0.4, 1e-3)
    qty_b = max(qty - qty_a, 1e-3)
    total_qty = qty_a + qty_b

    order = Order(
        order_id="ORD-1",
        symbol="EURUSD",
        side="BUY",
        quantity=total_qty,
        order_type=OrderType.MARKET,
        created_at=datetime.utcnow(),
    )

    order.add_fill(qty_a, first_price)
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.is_active is True

    order.add_fill(qty_b, second_price)

    expected_average = ((qty_a * first_price) + (qty_b * second_price)) / total_qty
    assert order.average_price == pytest.approx(expected_average)
    assert order.status is OrderStatus.FILLED
    assert order.is_active is False
    assert order.filled_quantity == pytest.approx(total_qty)


@given(price_floats, st.floats(min_value=0.05, max_value=0.95))
def test_partial_then_final_fill_sets_timestamps(price: float, ratio: float) -> None:
    total_qty = 1000.0
    first_leg = total_qty * ratio
    second_leg = total_qty - first_leg

    order = Order(
        order_id="ORD-2",
        symbol="GBPUSD",
        side="SELL",
        quantity=total_qty,
        order_type=OrderType.LIMIT,
        price=price,
    )

    order.add_fill(first_leg, price)
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_at is None

    order.add_fill(second_leg, price)
    assert order.status is OrderStatus.FILLED
    assert order.filled_at is not None


@given(st.sampled_from(list(OrderStatus)))
def test_update_status_controls_is_active(status: OrderStatus) -> None:
    order = Order(
        order_id="ORD-3",
        symbol="USDJPY",
        side="BUY",
        quantity=1.0,
        order_type=OrderType.MARKET,
    )

    order.update_status(status)
    assert order.is_active is (status in {OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED})
