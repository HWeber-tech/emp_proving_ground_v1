import asyncio
from typing import Any, Awaitable

import pytest

from src.core.event_bus import AsyncEventBus, Event, TopicBus


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


@pytest.mark.asyncio()
async def test_event_bus_default_supervisor_tracks_worker_tasks() -> None:
    bus = AsyncEventBus()
    await bus.start()
    supervisor: Any | None = None
    try:
        supervisor = getattr(bus, "_task_supervisor", None)
        assert supervisor is not None, "Default event bus should attach a task supervisor"
        assert supervisor.active_count >= 1

        handler_fired = asyncio.Event()

        async def _handler(event: Event) -> None:
            handler_fired.set()

        bus.subscribe("runtime.default", _handler)
        await bus.publish(Event(type="runtime.default", payload={}))
        await asyncio.wait_for(handler_fired.wait(), timeout=1.0)

        descriptions = supervisor.describe()
        assert any(
            entry.get("metadata", {}).get("task") == "worker" for entry in descriptions
        )
    finally:
        await bus.stop()

    assert supervisor is not None
    assert supervisor.active_count == 0


@pytest.mark.asyncio()
async def test_event_bus_task_snapshots_expose_worker_metadata() -> None:
    bus = AsyncEventBus()
    await bus.start()
    try:
        snapshots = bus.task_snapshots()
        assert snapshots, "Expected worker task snapshot when bus is running"
        assert any(
            entry.get("metadata", {}).get("task") == "worker" for entry in snapshots
        )
    finally:
        await bus.stop()


@pytest.mark.asyncio()
async def test_topic_bus_task_snapshots_delegate() -> None:
    bus = AsyncEventBus()
    topic_bus = TopicBus(bus)
    await bus.start()
    try:
        assert topic_bus.task_snapshots() == bus.task_snapshots()
    finally:
        await bus.stop()


@pytest.mark.asyncio()
async def test_event_bus_cancels_supervised_tasks_when_swapping_factory() -> None:
    bus = AsyncEventBus()
    await bus.start()
    blocker = asyncio.Event()
    cancelled = asyncio.Event()
    started = asyncio.Event()
    try:
        async def _handler(event: Event) -> None:
            try:
                started.set()
                await blocker.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        bus.subscribe("runtime.swap", _handler)
        await bus.publish(Event(type="runtime.swap", payload={}))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        factory = _RecordingFactory()
        supervisor = getattr(bus, "_task_supervisor", None)
        assert supervisor is not None
        assert supervisor.active_count >= 1
        metadata_entries = supervisor.describe()
        assert any(
            entry.get("metadata", {}).get("task") == "handler"
            for entry in metadata_entries
        )
        bus.set_task_factory(factory)

        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    finally:
        blocker.set()
        await bus.stop()
