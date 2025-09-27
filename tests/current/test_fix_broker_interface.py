from __future__ import annotations

import asyncio
from typing import Any

import pytest
import simplefix

from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


@pytest.mark.asyncio
async def test_execution_report_emits_fill_details() -> None:
    bus = _StubEventBus()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    broker = FIXBrokerInterface(bus, queue, fix_initiator=None)

    captured: dict[str, Any] = {}

    def _on_fill(order_id: str, payload: dict[str, Any]) -> None:
        captured["order_id"] = order_id
        captured["payload"] = payload.copy()

    broker.add_event_listener("filled", _on_fill)

    msg = simplefix.FixMessage()
    msg.append_pair(11, "ORD-123")
    msg.append_pair(150, "2")
    msg.append_pair(32, "2")
    msg.append_pair(31, "1.25")
    msg.append_pair(14, "5")
    msg.append_pair(151, "3")

    await broker._handle_execution_report(msg)

    assert captured["order_id"] == "ORD-123"
    payload = captured["payload"]
    assert payload["last_qty"] == pytest.approx(2.0)
    assert payload["last_px"] == pytest.approx(1.25)
    assert payload["cum_qty"] == pytest.approx(5.0)
    assert payload["leaves_qty"] == pytest.approx(3.0)
    assert payload["filled_qty"] == pytest.approx(5.0)

    assert bus.events
    event_type, event_payload = bus.events[-1]
    assert event_type == "order_update"
    assert event_payload["last_qty"] == pytest.approx(2.0)
    assert event_payload["cum_qty"] == pytest.approx(5.0)
