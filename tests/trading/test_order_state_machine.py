from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trading.order_management import (
    OrderExecutionEvent,
    OrderMetadata,
    OrderStateError,
    OrderStateMachine,
    OrderStatus,
)


def _ts() -> datetime:
    return datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_order_state_machine_full_fill_flow() -> None:
    machine = OrderStateMachine()
    machine.register_order(OrderMetadata("ORD1", "EURUSD", "BUY", 10))

    state = machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD1",
            event_type="acknowledged",
            exec_type="0",
            timestamp=_ts(),
        )
    )
    assert state.status is OrderStatus.ACKNOWLEDGED

    state = machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD1",
            event_type="partial_fill",
            exec_type="1",
            last_quantity=4,
            last_price=101.0,
            cumulative_quantity=4,
            timestamp=_ts(),
        )
    )
    assert state.status is OrderStatus.PARTIALLY_FILLED
    assert pytest.approx(state.filled_quantity) == 4
    assert pytest.approx(state.average_fill_price or 0.0) == 101.0

    state = machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD1",
            event_type="filled",
            exec_type="2",
            last_quantity=6,
            last_price=102.0,
            cumulative_quantity=10,
            timestamp=_ts(),
        )
    )
    assert state.status is OrderStatus.FILLED
    assert pytest.approx(state.filled_quantity) == 10
    assert pytest.approx(state.remaining_quantity, abs=1e-9) == 0
    # Weighted average of the two fills
    assert pytest.approx(state.average_fill_price or 0.0, rel=1e-9) == pytest.approx(
        (4 * 101 + 6 * 102) / 10
    )


def test_order_state_machine_rejects_invalid_progression() -> None:
    machine = OrderStateMachine()
    machine.register_order(OrderMetadata("ORD2", "GBPUSD", "SELL", 5))

    # Cannot overfill an order
    machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD2",
            event_type="partial_fill",
            exec_type="1",
            last_quantity=2,
            last_price=1.2,
            timestamp=_ts(),
        )
    )

    with pytest.raises(OrderStateError):
        machine.apply_event(
            OrderExecutionEvent(
                order_id="ORD2",
                event_type="filled",
                exec_type="2",
                last_quantity=4,
                cumulative_quantity=6,
                last_price=1.2,
                timestamp=_ts(),
            )
        )


def test_order_state_machine_reject_transition_requires_pre_terminal_state() -> None:
    machine = OrderStateMachine()
    machine.register_order(OrderMetadata("ORD3", "AAPL", "BUY", 1))

    machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD3",
            event_type="acknowledged",
            exec_type="0",
            timestamp=_ts(),
        )
    )

    machine.apply_event(
        OrderExecutionEvent(
            order_id="ORD3",
            event_type="cancelled",
            exec_type="4",
            timestamp=_ts(),
        )
    )

    with pytest.raises(OrderStateError):
        machine.apply_event(
            OrderExecutionEvent(
                order_id="ORD3",
                event_type="rejected",
                exec_type="8",
                timestamp=_ts(),
            )
        )

