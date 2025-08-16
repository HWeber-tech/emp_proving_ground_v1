import asyncio
import threading
import pytest

from src.core.event_bus import AsyncEventBus, Event


@pytest.mark.asyncio
async def test_publish_from_sync_threadsafe():
    bus = AsyncEventBus()
    await bus.start()
    try:
        seen = asyncio.Event()
        received: list[dict] = []

        async def handler(ev: Event) -> None:
            received.append(ev.payload)
            seen.set()

        bus.subscribe("sync.topic", handler)

        results: list[int | None] = []

        def worker():
            res = bus.publish_from_sync(Event(type="sync.topic", payload={"z": 9}))
            results.append(res)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Ensure the handler fired without raising
        await asyncio.wait_for(seen.wait(), timeout=1.0)
        assert received == [{"z": 9}]
        assert results and results[0] == 1
    finally:
        await bus.stop()