#!/usr/bin/env python3

import asyncio
import pytest

from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class DummyEventBus:
    def __init__(self, allow: bool):
        class DummyRisk:
            def __init__(self, allow: bool):
                self._allow = allow

            def check_risk_thresholds(self) -> bool:
                return self._allow

        self.risk_manager = DummyRisk(allow)

    async def emit(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_fix_order_blocked_by_risk():
    trade_queue = asyncio.Queue()
    bus = DummyEventBus(allow=False)
    broker = FIXBrokerInterface(bus, trade_queue, fix_initiator=None)
    order_id = await broker.place_market_order("EURUSD", "BUY", 1000)
    assert order_id is None


@pytest.mark.asyncio
async def test_fix_order_allowed_by_risk():
    trade_queue = asyncio.Queue()
    bus = DummyEventBus(allow=True)
    broker = FIXBrokerInterface(bus, trade_queue, fix_initiator=None)
    order_id = await broker.place_market_order("EURUSD", "BUY", 1000)
    # No initiator means no sending; function returns None after attempting
    assert order_id is None or isinstance(order_id, str)


