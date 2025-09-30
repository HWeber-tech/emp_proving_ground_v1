import asyncio
from collections.abc import Mapping

import pytest

from src.operational.health_monitor import HealthMonitor


class _DummyStateStore:
    def __init__(self) -> None:
        self.data: dict[str, tuple[str, int | None]] = {}

    async def set(self, key: str, value: str, expire: int | None = None) -> None:
        self.data[key] = (value, expire)

    async def get(self, key: str) -> str | None:
        record = self.data.get(key)
        return record[0] if record else None


class _DummyEventBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, Mapping[str, object]]] = []

    async def emit(self, event_name: str, payload: Mapping[str, object]) -> None:
        self.emitted.append((event_name, payload))


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        coro,
        *,
        name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        task = asyncio.create_task(coro)
        self.calls.append({"task": task, "name": name, "metadata": metadata})
        return task


class _MinimalSupervisor:
    def __init__(self) -> None:
        self.names: list[str | None] = []

    def create(self, coro, *, name: str | None = None):  # pragma: no cover - compatibility path
        task = asyncio.create_task(coro)
        self.names.append(name)
        return task


@pytest.mark.asyncio
async def test_health_monitor_runs_under_task_supervision():
    state_store = _DummyStateStore()
    event_bus = _DummyEventBus()
    supervisor = _RecordingSupervisor()

    monitor = HealthMonitor(state_store, event_bus, task_supervisor=supervisor)
    monitor.check_interval = 0.01

    await monitor.start()
    await asyncio.sleep(0.03)
    await monitor.stop()

    assert supervisor.calls, "health monitor should register a supervised task"
    stored_keys = list(state_store.data.keys())
    assert stored_keys, "health checks should be persisted"
    for call in supervisor.calls:
        assert call["metadata"] == {"component": "operational.health_monitor"}
        assert call["name"] == "health-monitor-loop"
        assert call["task"].done()


@pytest.mark.asyncio
async def test_health_monitor_falls_back_when_metadata_not_supported():
    state_store = _DummyStateStore()
    event_bus = _DummyEventBus()
    supervisor = _MinimalSupervisor()

    monitor = HealthMonitor(state_store, event_bus, task_supervisor=supervisor)
    monitor.check_interval = 0.01

    await monitor.start()
    await asyncio.sleep(0.03)
    await monitor.stop()

    assert supervisor.names == ["health-monitor-loop"]
