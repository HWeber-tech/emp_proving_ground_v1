import asyncio

import pytest

from src.core.event_bus import event_bus, get_global_bus


def test_topic_facade_compat_works_with_async_handler():
    bus = get_global_bus()

    done = asyncio.Event()
    received: list[tuple[str, dict]] = []

    async def handler(topic: str, payload):
        received.append((topic, payload))
        done.set()

    # Subscribe via facade
    handle = bus.subscribe_topic("bar", handler)

    # Publish via facade sync method
    count = bus.publish_sync("bar", {"x": 1})
    assert count == 1

    # Wait on the loop thread for async handler completion
    loop = event_bus._loop
    assert loop is not None and loop.is_running()
    fut = asyncio.run_coroutine_threadsafe(done.wait(), loop)
    fut.result(timeout=1.0)

    assert received == [("bar", {"x": 1})]

    # Cleanup
    bus.unsubscribe(handle)