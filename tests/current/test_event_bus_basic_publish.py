import asyncio

import pytest

from src.core.event_bus import AsyncEventBus, Event


@pytest.mark.asyncio
async def test_event_bus_basic_publish():
    bus = AsyncEventBus()
    await bus.start()
    try:
        received: list[dict] = []
        done = asyncio.Event()

        async def handler(ev: Event) -> None:
            received.append(ev.payload)
            done.set()

        bus.subscribe("foo", handler)

        await bus.publish(Event(type="foo", payload={"k": "v"}))
        await asyncio.wait_for(done.wait(), timeout=1.0)

        assert received == [{"k": "v"}]
    finally:
        await bus.stop()
