#!/usr/bin/env python3
"""
Safe paper-trade dry run via FIXBrokerInterface without sending real FIX messages.
- Requires RUN_MODE != live
- Simulates a market order and an execution report
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.governance.system_config import SystemConfig
from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class DummyInitiator:
    """Dummy initiator that captures messages instead of sending to broker."""
    def __init__(self):
        self.sent_messages = []

    def send_message(self, msg) -> bool:
        self.sent_messages.append(msg)
        return True  # Simulate successful send


class DummyEventBus:
    async def emit(self, event_name: str, payload):
        print(f"Event emitted: {event_name} -> {payload}")


async def main(symbol: str = "EURUSD", side: str = "BUY", qty: float = 0.01) -> int:
    cfg = SystemConfig()
    if str(cfg.run_mode).lower() == "live":
        print("Refusing to run in live mode. Set RUN_MODE=paper or mock.")
        return 1

    trade_queue: asyncio.Queue = asyncio.Queue()
    initiator = DummyInitiator()
    event_bus = DummyEventBus()

    broker = FIXBrokerInterface(event_bus, trade_queue, initiator)
    await broker.start()

    # Place paper order
    order_id = await broker.place_market_order(symbol=symbol, side=side, quantity=qty)
    if not order_id:
        print("Order placement simulation failed")
        return 1

    print(f"Paper order placed: {order_id} {side} {qty} {symbol}")

    # Simulate an execution report (fill)
    exec_report = {
        35: b"8",        # ExecutionReport
        11: order_id.encode("utf-8"),
        150: b"F",       # ExecType=Fill
    }
    await trade_queue.put(exec_report)

    # Allow handler to process the message
    await asyncio.sleep(0.5)

    status = broker.get_order_status(order_id)
    print("Order status:", status or {})

    await broker.stop()
    # Summary
    print(f"Messages captured by dummy initiator: {len(initiator.sent_messages)}")
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)



