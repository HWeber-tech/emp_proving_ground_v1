from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest
import simplefix

from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class DummyEventBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        self.emitted.append((topic, payload))


@pytest.mark.asyncio
async def test_fix_interface_emits_structured_order_events() -> None:
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    interface = FIXBrokerInterface(DummyEventBus(), trade_queue, fix_initiator=None)

    # Pre-populate with order metadata to mirror placement stage
    interface.orders["ORD-1"] = {
        "symbol": "EURUSD",
        "side": "BUY",
        "quantity": 10,
        "status": "PENDING",
        "timestamp": datetime.utcnow(),
    }

    observed: list[tuple[str, dict[str, Any]]] = []

    interface.add_event_listener(
        "acknowledged", lambda order_id, payload: observed.append((order_id, payload))
    )
    interface.add_event_listener(
        "partial_fill", lambda order_id, payload: observed.append((order_id, payload))
    )
    interface.add_event_listener(
        "filled", lambda order_id, payload: observed.append((order_id, payload))
    )

    # Ack message
    ack_msg = simplefix.FixMessage()
    ack_msg.append_pair(11, "ORD-1")
    ack_msg.append_pair(150, "0")
    await interface._handle_execution_report(ack_msg)  # type: ignore[attr-defined]

    # Partial fill
    partial_msg = simplefix.FixMessage()
    partial_msg.append_pair(11, "ORD-1")
    partial_msg.append_pair(150, "1")
    partial_msg.append_pair(32, "4")
    partial_msg.append_pair(31, "101.0")
    partial_msg.append_pair(14, "4")
    await interface._handle_execution_report(partial_msg)  # type: ignore[attr-defined]

    # Final fill
    fill_msg = simplefix.FixMessage()
    fill_msg.append_pair(11, "ORD-1")
    fill_msg.append_pair(150, "2")
    fill_msg.append_pair(32, "6")
    fill_msg.append_pair(31, "102.0")
    fill_msg.append_pair(14, "10")
    await interface._handle_execution_report(fill_msg)  # type: ignore[attr-defined]

    assert [event for event, _ in observed] == ["ORD-1", "ORD-1", "ORD-1"]
    statuses = [payload["status"] for _, payload in observed]
    assert statuses == ["ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED"]
    filled_qty = [payload.get("filled_qty") for _, payload in observed]
    assert filled_qty[-1] == pytest.approx(10)

