import asyncio
from typing import Any, Awaitable

import pytest

from src.core.event_bus import AsyncEventBus, Event


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        coro: Awaitable[object],
        *,
        name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> asyncio.Task[object]:
        task = asyncio.create_task(coro, name=name)
        payload: dict[str, object] = {"task": task, "name": name}
        if metadata is not None:
            payload["metadata"] = metadata
        self.calls.append(payload)
        return task


@pytest.mark.asyncio()
async def test_event_bus_uses_custom_task_factory_metadata() -> None:
    factory = _RecordingFactory()
    bus = AsyncEventBus(task_factory=factory)
    await bus.start()
    try:
        done = asyncio.Event()

        async def _handler(event: Event) -> None:
            done.set()

        bus.subscribe("telemetry.test", _handler)

        await bus.publish(Event(type="telemetry.test", payload={}))
        await asyncio.wait_for(done.wait(), timeout=1.0)
    finally:
        await bus.stop()

    worker_entries = [
        call for call in factory.calls if call.get("metadata", {}).get("task") == "worker"
    ]
    assert worker_entries

    handler_entries = [
        call for call in factory.calls if call.get("metadata", {}).get("task") == "handler"
    ]
    assert handler_entries
    assert handler_entries[0]["metadata"]["event_type"] == "telemetry.test"


@pytest.mark.asyncio()
async def test_event_bus_set_task_factory_before_start() -> None:
    factory = _RecordingFactory()
    bus = AsyncEventBus()
    bus.set_task_factory(factory)

    await bus.start()
    try:
        await asyncio.sleep(0)
    finally:
        await bus.stop()

    assert any(call.get("metadata", {}).get("task") == "worker" for call in factory.calls)
