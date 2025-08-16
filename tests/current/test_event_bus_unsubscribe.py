import asyncio
import pytest

from src.core.event_bus import AsyncEventBus, Event


@pytest.mark.asyncio
async def test_event_bus_unsubscribe():
    bus = AsyncEventBus()
    await bus.start()
    try:
        calls: list[dict] = []
        done = asyncio.Event()

        async def handler(ev: Event) -> None:
            calls.append(ev.payload)
            done.set()

        handle = bus.subscribe("foo", handler)

        # First publish should be delivered
        await bus.publish(Event(type="foo", payload={"n": 1}))
        await asyncio.wait_for(done.wait(), timeout=1.0)
        assert calls == [{"n": 1}]

        # Unsubscribe and verify no delivery on subsequent publish
        bus.unsubscribe(handle)
        calls.clear()
        done.clear()

        await bus.publish(Event(type="foo", payload={"n": 2}))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(done.wait(), timeout=0.2)
        assert calls == []
    finally:
        await bus.stop()