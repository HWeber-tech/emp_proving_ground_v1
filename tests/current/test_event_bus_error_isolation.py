import asyncio
import logging

import pytest

from src.core.event_bus import AsyncEventBus, Event, get_global_bus


@pytest.mark.asyncio
async def test_async_publish_error_isolation(caplog):
    bus = AsyncEventBus()
    await bus.start()

    latch = asyncio.Event()

    def bad_sync(ev: Event):
        raise ValueError("boom-sync")

    async def bad_async(ev: Event):
        raise RuntimeError("boom-async")

    async def good(ev: Event):
        latch.set()

    h1 = bus.subscribe("iso.async", bad_sync)
    h2 = bus.subscribe("iso.async", bad_async)
    h3 = bus.subscribe("iso.async", good)

    caplog.set_level(logging.ERROR, logger="src.core.event_bus")
    try:
        await bus.publish(Event(type="iso.async", payload={"k": "v"}))
        await asyncio.wait_for(latch.wait(), timeout=2.0)

        errors = [rec for rec in caplog.records if "Error in handler" in rec.getMessage()]
        assert len(errors) == 2
    finally:
        bus.unsubscribe(h1)
        bus.unsubscribe(h2)
        bus.unsubscribe(h3)
        await bus.stop()


def test_topicbus_publish_sync_error_isolation(caplog):
    bus = get_global_bus()

    sink: list[int] = []

    def bad(_type: str, payload: int):
        raise RuntimeError("boom-sync-topic")

    def good(_type: str, payload: int):
        sink.append(payload)

    h_bad = bus.subscribe_topic("iso.sync", bad)
    h_good = bus.subscribe_topic("iso.sync", good)

    caplog.set_level(logging.ERROR, logger="src.core.event_bus")
    try:
        count = bus.publish_sync("iso.sync", 7)
        assert isinstance(count, int) and count >= 1
        assert sink == [7]

        errors = [rec for rec in caplog.records if "Error in handler" in rec.getMessage()]
        assert len(errors) >= 1
    finally:
        bus.unsubscribe(h_bad)
        bus.unsubscribe(h_good)