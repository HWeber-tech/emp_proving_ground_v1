from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime

from src.trading.models.order import Order, OrderStatus


def _make_min_order(**overrides):
    return Order(
        order_id="o-1",
        symbol="TEST",
        side="BUY",
        quantity=10.0,
        order_type="LIMIT",
        price=100.0,
        **overrides,
    )


def test_construction_defaults_and_timestamps():
    o = _make_min_order()
    # status default (if present)
    assert hasattr(o, "status")
    assert o.status == OrderStatus.PENDING

    # created_at / updated_at are datetime and updated_at >= created_at
    assert isinstance(o.created_at, datetime)
    assert isinstance(o.updated_at, datetime)
    assert o.updated_at >= o.created_at


def test_equality_semantics_with_explicit_timestamps():
    ts = datetime(2024, 1, 1, 12, 0, 0)

    o1 = _make_min_order(created_at=ts, updated_at=ts, status=OrderStatus.PENDING)
    o2 = _make_min_order(created_at=ts, updated_at=ts, status=OrderStatus.PENDING)
    assert o1 == o2

    # change a field (e.g., price) -> inequality
    o2.price = 101.0
    assert o1 != o2


def test_state_transitions_and_flags():
    o = _make_min_order()
    # initial flags
    if hasattr(o, "is_active"):
        assert o.is_active is True
    if hasattr(o, "is_filled"):
        assert o.is_filled is False

    prev_updated = o.updated_at
    assert prev_updated is not None
    assert hasattr(o, "update_status")
    o.update_status(OrderStatus.REJECTED)

    assert o.status == OrderStatus.REJECTED
    assert o.updated_at is not None
    assert o.updated_at >= prev_updated

    if hasattr(o, "is_active"):
        assert o.is_active is False
    if hasattr(o, "is_filled"):
        assert o.is_filled is False


def test_partial_and_complete_fills_weighted_average():
    o = _make_min_order()

    prev_updated = o.updated_at
    assert prev_updated is not None
    assert hasattr(o, "add_fill")
    o.add_fill(4, 100.0)

    assert o.status == OrderStatus.PARTIALLY_FILLED
    assert o.filled_quantity == 4
    assert math.isclose(o.average_price or 0.0, 100.0, rel_tol=1e-9, abs_tol=1e-12)
    assert o.updated_at is not None
    assert o.updated_at >= prev_updated

    prev_updated = o.updated_at
    assert prev_updated is not None
    o.add_fill(6, 102.0)

    assert o.status == OrderStatus.FILLED
    assert o.filled_quantity == 10
    assert math.isclose(o.average_price or 0.0, 101.2, rel_tol=1e-9, abs_tol=1e-12)
    assert o.filled_at is not None
    assert o.updated_at is not None
    assert o.updated_at >= prev_updated


def test_serialization_roundtrip():
    ts = datetime(2024, 1, 1, 12, 0, 0)
    original = _make_min_order(created_at=ts, updated_at=ts, status=OrderStatus.PENDING)

    d = asdict(original)
    # reconstruct with explicit timestamps/status to avoid drift
    roundtrip = Order(**d)

    assert original == roundtrip