import asyncio
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.operational.health_monitor import HealthMonitor, SupportsTaskSupervision


class RecordingSupervisor(SupportsTaskSupervision):
    """Minimal task supervisor used to assert background registration."""

    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[Any]] = []
        self.metadata: list[dict[str, Any] | None] = []

    def create(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        self.metadata.append(dict(metadata) if metadata is not None else None)
        return task


class InMemoryStateStore:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def set(self, key: str, value: str, expire: int | None = None) -> bool:
        self.data[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)


class StubEventBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        self.emitted.append((topic, payload))


@pytest.mark.asyncio
async def test_health_monitor_registers_background_task_under_supervision() -> None:
    supervisor = RecordingSupervisor()
    state_store = InMemoryStateStore()
    event_bus = StubEventBus()
    monitor = HealthMonitor(
        state_store,
        event_bus,
        task_supervisor=supervisor,
        check_interval_seconds=0.01,
    )

    health_check = {
        "check_id": "test",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "status": "HEALTHY",
        "components": {},
    }
    monitor._perform_health_check = AsyncMock(return_value=health_check)  # type: ignore[method-assign]

    await monitor.start()
    await asyncio.sleep(0.02)
    assert supervisor.tasks, "Health monitor should create a supervised task"
    assert supervisor.metadata[-1] == {"component": "health-monitor"}

    await monitor.stop()
    for task in supervisor.tasks:
        assert task.done()


@pytest.mark.asyncio
async def test_health_monitor_emits_critical_events() -> None:
    supervisor = RecordingSupervisor()
    state_store = InMemoryStateStore()
    event_bus = StubEventBus()
    monitor = HealthMonitor(
        state_store,
        event_bus,
        task_supervisor=supervisor,
        check_interval_seconds=0.01,
    )

    checks = iter(
        [
            {
                "check_id": "critical",
                "timestamp": "2024-01-01T00:00:00+00:00",
                "status": "CRITICAL",
                "components": {},
            },
            {
                "check_id": "healthy",
                "timestamp": "2024-01-01T00:00:01+00:00",
                "status": "HEALTHY",
                "components": {},
            },
        ]
    )

    async def fake_perform() -> dict[str, Any]:
        try:
            return next(checks)
        except StopIteration:
            return {
                "check_id": "steady",
                "timestamp": "2024-01-01T00:00:02+00:00",
                "status": "HEALTHY",
                "components": {},
            }

    monitor._perform_health_check = fake_perform  # type: ignore[method-assign]

    await monitor.start()
    await asyncio.sleep(0.03)
    await monitor.stop()

    assert ("health_critical", event_bus.emitted[0][1]) in event_bus.emitted
    assert monitor.health_history, "Health checks should be persisted"
