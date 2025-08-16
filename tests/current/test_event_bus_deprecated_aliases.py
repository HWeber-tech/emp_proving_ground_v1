import asyncio
import pytest

from src.core.event_bus import AsyncEventBus, Event, TopicBus, get_global_bus


@pytest.mark.asyncio
async def test_async_emit_deprecated_and_delivers():
    bus = AsyncEventBus()
    await bus.start()

    latch = asyncio.Event()
    received = {}

    async def handler(ev: Event):
        if ev.type == "test.deprecated.emit" and ev.payload == {"x": 1}:
            received["ok"] = True
            latch.set()

    handle = bus.subscribe("test.deprecated.emit", handler)
    try:
        with pytest.warns(DeprecationWarning):
            await bus.emit("test.deprecated.emit", {"x": 1})
        await asyncio.wait_for(latch.wait(), timeout=2.0)
        assert received.get("ok") is True
    finally:
        bus.unsubscribe(handle)
        await bus.stop()


def test_topicbus_publish_deprecated_alias_and_sync_fanout(monkeypatch):
    # Ensure deterministic deprecation warning behavior
    monkeypatch.setattr(TopicBus, "_warned_publish_once", False, raising=False)

    bus = get_global_bus()

    sink: list[int] = []

    def handler(_type: str, payload):
        sink.append(payload)

    handle = bus.subscribe_topic("test.deprecated.publish", handler)
    try:
        with pytest.warns(DeprecationWarning):
            bus.publish("test.deprecated.publish", 1234)
        # publish() delegates to publish_sync(), so this should be immediate
        assert sink == [1234]
    finally:
        bus.unsubscribe(handle)