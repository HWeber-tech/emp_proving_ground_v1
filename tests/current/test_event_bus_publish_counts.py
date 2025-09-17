import asyncio
import logging

import pytest

from src.core.event_bus import AsyncEventBus, Event, get_global_bus


def test_topicbus_publish_sync_returns_count():
    bus = get_global_bus()

    sink: list[tuple[str, int]] = []

    def h1(_type: str, payload):
        sink.append(("h1", payload))

    def h2(_type: str, payload):
        sink.append(("h2", payload))

    h1_handle = bus.subscribe_topic("topic.sync.count", h1)
    h2_handle = bus.subscribe_topic("topic.sync.count", h2)
    try:
        count = bus.publish_sync("topic.sync.count", {"n": 1})
        assert isinstance(count, int)
        assert count == 2
        # Synchronous fan-out: both handlers executed immediately
        labels = {name for name, _ in sink}
        assert labels == {"h1", "h2"}
    finally:
        bus.unsubscribe(h1_handle)
        bus.unsubscribe(h2_handle)


def test_publish_from_sync_not_running_warns_and_returns_none(caplog):
    bus = AsyncEventBus()
    caplog.set_level(logging.WARNING, logger="src.core.event_bus")

    res = bus.publish_from_sync(Event(type="not.running", payload=1))
    assert res is None
    assert any(
        "publish_from_sync called while loop/bus not running" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_publish_from_sync_running_returns_count():
    bus = AsyncEventBus()
    await bus.start()

    latch = asyncio.Event()

    async def handler(ev: Event):
        if ev.type == "sync.running" and ev.payload == 42:
            latch.set()

    handle = bus.subscribe("sync.running", handler)
    try:
        count = bus.publish_from_sync(Event(type="sync.running", payload=42))
        assert count == 1
        await asyncio.wait_for(latch.wait(), timeout=2.0)
    finally:
        bus.unsubscribe(handle)
        await bus.stop()
