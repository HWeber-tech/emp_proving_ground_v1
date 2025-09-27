from __future__ import annotations

import asyncio

import pytest

from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class _StubMessage:
    def __init__(self, data: dict[int, object]) -> None:
        self._data = data

    def get(self, tag: int) -> object | None:
        return self._data.get(tag)


class _StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    async def emit(self, name: str, payload: dict[str, object]) -> None:
        self.events.append((name, payload))


@pytest.mark.asyncio
async def test_execution_report_updates_state_machine() -> None:
    interface = FIXBrokerInterface(_StubBus(), asyncio.Queue(), fix_initiator=None)

    message = _StubMessage(
        {
            11: b"ORDER1",
            150: b"1",
            32: b"4",
            31: b"1.25",
            55: b"EURUSD",
            54: b"1",
            38: b"10",
            151: b"6",
        }
    )

    await interface._handle_execution_report(message)

    status = interface.get_order_status("ORDER1")
    assert status is not None
    assert status["status"] == "PARTIALLY_FILLED"
    assert status["filled_quantity"] == pytest.approx(4.0)
    assert status["remaining_quantity"] == pytest.approx(6.0)
    assert status["symbol"] == "EURUSD"
    assert status["side"] == "BUY"


@pytest.mark.asyncio
async def test_execution_report_final_fill_and_event_bus() -> None:
    bus = _StubBus()
    interface = FIXBrokerInterface(bus, asyncio.Queue(), fix_initiator=None)

    partial = _StubMessage(
        {
            11: b"ORDER2",
            150: b"1",
            32: b"5",
            31: b"1.10",
            55: b"AAPL",
            54: b"2",
            38: b"10",
            151: b"5",
        }
    )
    await interface._handle_execution_report(partial)

    fill = _StubMessage(
        {
            11: b"ORDER2",
            150: b"2",
            32: b"5",
            31: b"1.12",
            55: b"AAPL",
            54: b"2",
            151: b"0",
        }
    )
    await interface._handle_execution_report(fill)

    status = interface.get_order_status("ORDER2")
    assert status is not None
    assert status["status"] == "FILLED"
    assert status["filled_quantity"] == pytest.approx(10.0)
    assert status["remaining_quantity"] == pytest.approx(0.0)
    assert bus.events, "Expected order_update events to be emitted"
    assert all(name == "order_update" for name, _ in bus.events)
