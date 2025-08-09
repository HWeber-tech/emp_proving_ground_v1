#!/usr/bin/env python3

import asyncio
import pytest

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.trading.integration.fix_broker_interface import FIXBrokerInterface


@pytest.mark.asyncio
async def test_fix_connection_manager_bridging(monkeypatch):
    # Create config (paper mode)
    cfg = SystemConfig()

    # Instantiate manager but do not start real sessions
    manager = FIXConnectionManager(cfg)

    # Prepare queues and components
    price_queue = asyncio.Queue()
    trade_queue = asyncio.Queue()

    # Inject queues via applications
    price_app = manager.get_application("price")
    trade_app = manager.get_application("trade")
    assert price_app and trade_app
    price_app.set_message_queue(price_queue)
    trade_app.set_message_queue(trade_queue)

    sensory = FIXSensoryOrgan(event_bus=None, price_queue=price_queue, config=cfg)
    broker = FIXBrokerInterface(event_bus=None, trade_queue=trade_queue, fix_initiator=None)

    # Start processors (they read from queues)
    await sensory.start()
    await broker.start()

    # Simulate bridged market data snapshot message as adapter would provide
    snapshot = {
        35: b"W",
        55: b"EURUSD",
        b"entries": [
            {"type": b"0", "px": 1.1000, "size": 100000.0},  # bid
            {"type": b"1", "px": 1.1002, "size": 100000.0},  # ask
        ],
    }
    await price_queue.put(snapshot)

    # Give the sensory organ a tick to process
    await asyncio.sleep(0.05)
    md = sensory.get_market_data("EURUSD")
    assert md.get("bid") == 1.1000
    assert md.get("ask") == 1.1002

    # Simulate execution report
    exec_report = {
        35: b"8",
        11: b"ORD_123",
        150: b"2",  # FILL
    }
    await trade_queue.put(exec_report)

    # Give the broker interface a tick to process
    await asyncio.sleep(0.05)
    # Broker stores orders; status remains PENDING until initiator responses, so just assert no crash
    await broker.stop()
    await sensory.stop()


