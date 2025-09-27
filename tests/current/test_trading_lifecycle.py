from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trading.order_management import (
    OrderEventType,
    OrderLifecycle,
    OrderStateError,
    OrderStatus,
    PositionTracker,
)


def _ts(hour: int) -> datetime:
    return datetime(2025, 1, 1, hour, tzinfo=timezone.utc)


# --- Order lifecycle ---------------------------------------------------------

def test_order_lifecycle_partial_fill_flow() -> None:
    lifecycle = OrderLifecycle("ORD-1", quantity=10, symbol="EURUSD", side="BUY", created_at=_ts(9))

    snapshot = lifecycle.apply_event(OrderEventType.ACKNOWLEDGED, timestamp=_ts(9))
    assert snapshot.status is OrderStatus.ACKNOWLEDGED

    snapshot = lifecycle.apply_event(
        OrderEventType.PARTIAL_FILL,
        timestamp=_ts(10),
        quantity=4,
        price=1.1,
        leaves_quantity=6,
    )
    assert snapshot.status is OrderStatus.PARTIALLY_FILLED
    assert snapshot.filled_quantity == pytest.approx(4.0)
    assert snapshot.remaining_quantity == pytest.approx(6.0)
    assert snapshot.average_price == pytest.approx(1.1)

    snapshot = lifecycle.apply_event(
        OrderEventType.PARTIAL_FILL,
        timestamp=_ts(11),
        quantity=3,
        price=1.2,
        leaves_quantity=3,
    )
    assert snapshot.status is OrderStatus.PARTIALLY_FILLED
    assert snapshot.filled_quantity == pytest.approx(7.0)
    assert snapshot.remaining_quantity == pytest.approx(3.0)
    assert snapshot.average_price == pytest.approx((4 * 1.1 + 3 * 1.2) / 7)

    snapshot = lifecycle.apply_event(
        OrderEventType.FILL,
        timestamp=_ts(12),
        quantity=3,
        price=1.15,
        leaves_quantity=0,
    )
    assert snapshot.status is OrderStatus.FILLED
    assert snapshot.remaining_quantity == pytest.approx(0.0)
    assert snapshot.filled_quantity == pytest.approx(10.0)


def test_order_lifecycle_cancel_and_reject() -> None:
    lifecycle = OrderLifecycle("ORD-2", quantity=5, symbol="AAPL", side="SELL", created_at=_ts(9))
    lifecycle.apply_event(OrderEventType.ACKNOWLEDGED, timestamp=_ts(9))
    snapshot = lifecycle.apply_event(OrderEventType.CANCELLED, timestamp=_ts(10), leaves_quantity=5)

    assert snapshot.status is OrderStatus.CANCELLED
    assert snapshot.remaining_quantity == pytest.approx(5.0)

    rejected = OrderLifecycle("ORD-3", quantity=2, symbol="MSFT", side="BUY", created_at=_ts(9))
    snapshot = rejected.apply_event(OrderEventType.REJECTED, timestamp=_ts(9), reason="NoLiquidity")
    assert snapshot.status is OrderStatus.REJECTED
    assert snapshot.remaining_quantity == pytest.approx(0.0)


def test_order_lifecycle_invalid_transition() -> None:
    lifecycle = OrderLifecycle("ORD-4", quantity=1, symbol="BTCUSD", side="BUY", created_at=_ts(9))

    lifecycle.apply_event(OrderEventType.ACKNOWLEDGED, timestamp=_ts(9))

    with pytest.raises(OrderStateError):
        lifecycle.apply_event(OrderEventType.PARTIAL_FILL, quantity=2, timestamp=_ts(10), price=100.0)


@pytest.mark.parametrize(
    "exec_type,expected_status",
    [
        ("0", OrderStatus.ACKNOWLEDGED),
        ("1", OrderStatus.PARTIALLY_FILLED),
        ("2", OrderStatus.FILLED),
        ("4", OrderStatus.CANCELLED),
        ("8", OrderStatus.REJECTED),
    ],
)
def test_order_lifecycle_apply_fix_execution(exec_type: str, expected_status: OrderStatus) -> None:
    lifecycle = OrderLifecycle("ORD-5", quantity=10, symbol="EURUSD", side="BUY", created_at=_ts(9))

    snapshot = lifecycle.apply_fix_execution(
        exec_type,
        last_qty=5 if exec_type in {"1", "2"} else 0,
        last_px=1.11,
        leaves_qty=5 if exec_type == "1" else 0,
        timestamp=_ts(10),
    )

    assert snapshot.status is expected_status


# --- Position tracker --------------------------------------------------------

def test_position_tracker_fifo_accounting() -> None:
    tracker = PositionTracker(pnl_mode="fifo")

    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))
    tracker.record_fill("AAPL", 5, 110.0, timestamp=_ts(10))
    tracker.update_mark_price("AAPL", 115.0, timestamp=_ts(11))
    tracker.record_fill("AAPL", -8, 120.0, timestamp=_ts(12))

    snapshot = tracker.get_position_snapshot("AAPL")

    assert snapshot.net_quantity == pytest.approx(7.0)
    assert snapshot.realized_pnl == pytest.approx(160.0)
    assert snapshot.unrealized_pnl == pytest.approx(55.0)
    assert snapshot.exposure == pytest.approx(805.0)
    assert snapshot.market_value == pytest.approx(805.0)


def test_position_tracker_lifo_changes_realized_pnl() -> None:
    tracker = PositionTracker(pnl_mode="lifo")

    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))
    tracker.record_fill("AAPL", 5, 110.0, timestamp=_ts(10))
    tracker.record_fill("AAPL", -8, 120.0, timestamp=_ts(11))

    snapshot = tracker.get_position_snapshot("AAPL")

    assert snapshot.net_quantity == pytest.approx(7.0)
    assert snapshot.realized_pnl == pytest.approx(110.0)
    assert snapshot.market_price == pytest.approx(120.0)
    assert snapshot.unrealized_pnl == pytest.approx(140.0)


def test_position_tracker_short_covering_flow() -> None:
    tracker = PositionTracker()

    tracker.record_fill("MSFT", -5, 50.0, timestamp=_ts(9))
    tracker.update_mark_price("MSFT", 45.0, timestamp=_ts(10))
    tracker.record_fill("MSFT", 3, 40.0, timestamp=_ts(11))

    snapshot = tracker.get_position_snapshot("MSFT")

    assert snapshot.net_quantity == pytest.approx(-2.0)
    assert snapshot.realized_pnl == pytest.approx(30.0)
    assert snapshot.unrealized_pnl == pytest.approx(10.0)
    assert snapshot.exposure == pytest.approx(90.0)


def test_position_tracker_reconciliation_and_reset() -> None:
    tracker = PositionTracker()
    tracker.record_fill("AAPL", 10, 100.0, timestamp=_ts(9))

    report = tracker.generate_reconciliation_report({"AAPL": 8.0, "MSFT": 1.0})

    assert report.has_discrepancies() is True
    assert {(diff.symbol, round(diff.difference, 6)) for diff in report.differences} == {
        ("AAPL", 2.0),
        ("MSFT", -1.0),
    }

    tracker.record_fill("AAPL", 1, 100.0, account="alpha", timestamp=_ts(9))
    tracker.record_fill("MSFT", 1, 50.0, account="beta", timestamp=_ts(10))

    assert tracker.accounts() == ("PRIMARY", "alpha", "beta")
    assert tracker.total_exposure() > 0

    tracker.reset()

    assert tracker.accounts() == ()
    assert tracker.total_exposure() == 0.0
