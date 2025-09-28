#!/usr/bin/env python3
"""Safe paper-trading dry run that mirrors the production execution pipeline."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from typing import Any

import simplefix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.governance.system_config import SystemConfig
from src.operational.structured_logging import get_logger, order_logging_context
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.trading.order_management import (
    InMemoryOrderEventJournal,
    OrderLifecycleProcessor,
    OrderMetadata,
    PositionTracker,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class DummyInitiator:
    """Dummy initiator that captures messages instead of sending to broker."""

    sent_messages: list[simplefix.FixMessage]

    def __init__(self) -> None:
        self.sent_messages = []

    def send_message(self, msg: simplefix.FixMessage) -> bool:
        self.sent_messages.append(msg)
        logger.info("dummy_initiator_send", message=str(msg))
        return True  # Simulate successful send


class LoggingEventBus:
    """Minimal event bus that mirrors production structured logging."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, payload))
        order_id = payload.get("order_id") or "unknown"
        with order_logging_context(order_id, event=event_name):
            logger.info("event_bus_emit", payload=payload)


def _execution_report(
    order_id: str,
    *,
    exec_type: str,
    last_qty: float | None = None,
    last_px: float | None = None,
    cum_qty: float | None = None,
) -> simplefix.FixMessage:
    message = simplefix.FixMessage()
    message.append_pair(35, "8")
    message.append_pair(11, order_id)
    message.append_pair(150, exec_type)
    if last_qty is not None:
        message.append_pair(32, f"{last_qty}")
    if last_px is not None:
        message.append_pair(31, f"{last_px}")
    if cum_qty is not None:
        message.append_pair(14, f"{cum_qty}")
    return message


async def main(symbol: str = "EURUSD", side: str = "BUY", qty: float = 0.01) -> int:
    cfg = SystemConfig()
    if str(cfg.run_mode).lower() == "live":
        logger.error("paper_trade_refused", reason="run_mode_live")
        return 1

    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    initiator = DummyInitiator()
    event_bus = LoggingEventBus()

    broker = FIXBrokerInterface(event_bus, trade_queue, initiator)
    journal = InMemoryOrderEventJournal()
    tracker = PositionTracker(default_account="PAPER")
    lifecycle = OrderLifecycleProcessor(journal=journal, position_tracker=tracker)

    def _log_lifecycle_event(order_id: str, payload: dict[str, Any]) -> None:
        with order_logging_context(order_id, event=payload.get("exec_type")) as log:
            log.info("broker_event", payload=payload)

    for event_type in ("acknowledged", "partial_fill", "filled", "cancelled", "rejected"):
        broker.add_event_listener(event_type, _log_lifecycle_event)

    lifecycle.attach_broker(broker)

    await broker.start()

    order_id = await broker.place_market_order(symbol=symbol, side=side, quantity=qty)
    if not order_id:
        logger.error("order_placement_failed", symbol=symbol, side=side, quantity=qty)
        await broker.stop()
        return 1

    metadata = OrderMetadata(
        order_id=order_id,
        symbol=symbol,
        side=side.upper(),
        quantity=qty,
        account="PAPER",
    )
    lifecycle.register_order(metadata)

    with order_logging_context(order_id, symbol=symbol, side=side.upper()) as log:
        log.info("paper_order_submitted", quantity=qty)

    fill_price = 1.2345
    await trade_queue.put(_execution_report(order_id, exec_type="0"))
    await trade_queue.put(
        _execution_report(
            order_id,
            exec_type="1",
            last_qty=qty / 2,
            last_px=fill_price,
            cum_qty=qty / 2,
        )
    )
    await trade_queue.put(
        _execution_report(
            order_id,
            exec_type="2",
            last_qty=qty / 2,
            last_px=fill_price,
            cum_qty=qty,
        )
    )

    await asyncio.sleep(0.5)

    snapshot = lifecycle.get_snapshot(order_id)
    with order_logging_context(order_id, status=snapshot.status.value) as log:
        log.info(
            "order_snapshot",
            filled_quantity=snapshot.filled_quantity,
            average_fill_price=snapshot.average_fill_price,
        )

    position = tracker.get_position_snapshot(symbol, account="PAPER")
    logger.info(
        "position_tracker_snapshot",
        symbol=position.symbol,
        net_quantity=position.net_quantity,
        realized_pnl=position.realized_pnl,
        unrealized_pnl=position.unrealized_pnl,
    )

    logger.info("journal_event_count", count=len(journal.records))
    broker_status = broker.get_order_status(order_id)
    logger.info("broker_order_state", state=broker_status)

    lifecycle.detach_broker()
    await broker.stop()

    logger.info(
        "paper_trade_complete",
        order_id=order_id,
        messages_sent=len(initiator.sent_messages),
        events_published=len(event_bus.events),
    )
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
