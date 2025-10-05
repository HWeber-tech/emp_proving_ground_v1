import logging
from dataclasses import dataclass

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from src.core.event_bus import EventBusStatistics
from src.operational.health_monitor import (
    HealthMonitor,
    _PsutilUnavailableError,
    _import_psutil,
)
from src.runtime.task_supervisor import TaskSupervisor


@dataclass
class _StubStatistics:
    running: bool = True
    loop_running: bool = True
    queue_size: int = 0
    queue_capacity: int | None = None
    subscriber_count: int = 0
    topic_subscribers: dict[str, int] | None = None
    published_events: int = 0
    dropped_events: int = 0
    handler_errors: int = 0
    last_event_timestamp: float | None = None
    last_error_timestamp: float | None = None
    started_at: float | None = None
    uptime_seconds: float | None = None

    def as_event_bus_stats(self) -> EventBusStatistics:
        return EventBusStatistics(
            running=self.running,
            loop_running=self.loop_running,
            queue_size=self.queue_size,
            queue_capacity=self.queue_capacity,
            subscriber_count=self.subscriber_count,
            topic_subscribers=self.topic_subscribers or {},
            published_events=self.published_events,
            dropped_events=self.dropped_events,
            handler_errors=self.handler_errors,
            last_event_timestamp=self.last_event_timestamp,
            last_error_timestamp=self.last_error_timestamp,
            started_at=self.started_at,
            uptime_seconds=self.uptime_seconds,
        )


class _StubEventBus:
    def __init__(self, stats: EventBusStatistics | Exception | None = None) -> None:
        self._stats = stats or _StubStatistics().as_event_bus_stats()

    def get_statistics(self) -> EventBusStatistics:
        if isinstance(self._stats, Exception):
            raise self._stats
        return self._stats


class _StubStateStore:
    def __init__(self, *, should_fail: bool = False) -> None:
        self._should_fail = should_fail
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, *, expire: int | None = None) -> None:
        if self._should_fail:
            raise RuntimeError("state store offline")
        self._store[key] = value

    async def get(self, key: str) -> str | None:
        if self._should_fail:
            raise RuntimeError("state store offline")
        return self._store.get(key)


def test_import_psutil_converts_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import(name: str) -> object:
        raise ImportError("missing dependency")

    monkeypatch.setattr("src.operational.health_monitor.import_module", _raise_import)

    with pytest.raises(_PsutilUnavailableError):
        _import_psutil()


@pytest.mark.asyncio
async def test_memory_probe_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus())

    def _raise_unavailable() -> object:
        raise _PsutilUnavailableError("psutil missing")

    monkeypatch.setattr("src.operational.health_monitor._import_psutil", _raise_unavailable)
    caplog.set_level(logging.WARNING)

    result = await monitor._check_memory()

    assert result["status"] == "UNKNOWN"
    assert "psutil missing" in result["error"]
    assert any("psutil unavailable for memory probe" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_memory_probe_logs_runtime_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus())

    class _FailingPsutil:
        @staticmethod
        def virtual_memory() -> object:
            raise RuntimeError("probe failed")

    monkeypatch.setattr("src.operational.health_monitor._import_psutil", lambda: _FailingPsutil())
    caplog.set_level(logging.ERROR)

    result = await monitor._check_memory()

    assert result["status"] == "UNKNOWN"
    assert "probe failed" in result["error"]
    assert any("Failed to sample memory usage" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_monitor_loop_logs_unhandled_errors(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus())
    monitor.check_interval = 0
    caplog.set_level(logging.ERROR)

    async def _fail_health_check() -> dict[str, object]:
        monitor.is_running = False
        raise RuntimeError("loop failure")

    async def _noop_store(_: dict[str, object]) -> None:
        return None

    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(monitor, "_perform_health_check", _fail_health_check)
    monkeypatch.setattr(monitor, "_store_health_check", _noop_store)
    monkeypatch.setattr("src.operational.health_monitor.asyncio.sleep", _fast_sleep)

    monitor.is_running = True
    await monitor._monitor_loop()

    assert any("Error in health monitoring loop" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_state_store_failure_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    monitor = HealthMonitor(_StubStateStore(should_fail=True), _StubEventBus())
    caplog.set_level(logging.WARNING)

    result = await monitor._check_state_store()

    assert result["status"] == "ERROR"
    assert "state store offline" in result["error"]
    assert any("State store health check failed" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_event_bus_statistics_snapshot() -> None:
    stats = _StubStatistics(
        running=True,
        loop_running=False,
        queue_size=5,
        subscriber_count=3,
        dropped_events=2,
    ).as_event_bus_stats()
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus(stats))

    result = await monitor._check_event_bus()

    assert result == {
        "status": "WARNING",
        "subscribers": 3,
        "queue_size": 5,
        "dropped_events": 2,
    }


@pytest.mark.asyncio
async def test_start_uses_task_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus())
    monitor.check_interval = 0
    supervisor = TaskSupervisor(namespace="test-health-monitor")

    recorded: dict[str, object] = {}
    original_create = TaskSupervisor.create

    def _recording_create(
        self: TaskSupervisor,
        coro: Coroutine[Any, Any, object],
        *,
        name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> asyncio.Task[object]:
        recorded["name"] = name
        recorded["metadata"] = metadata
        return original_create(self, coro, name=name, metadata=metadata)

    monkeypatch.setattr(TaskSupervisor, "create", _recording_create)

    async def _single_check() -> dict[str, object]:
        monitor.is_running = False
        return {"status": "HEALTHY"}

    async def _noop_store(_: dict[str, object]) -> None:
        return None

    monkeypatch.setattr(monitor, "_perform_health_check", _single_check)
    monkeypatch.setattr(monitor, "_store_health_check", _noop_store)

    await monitor.start(task_supervisor=supervisor)

    task = monitor._monitor_task
    if task is not None:
        await task

    assert recorded["name"] == "health-monitor-loop"
    assert recorded["metadata"] == {
        "component": "health-monitor",
        "interval_seconds": 0,
    }
    assert monitor._monitor_task is None


@pytest.mark.asyncio
async def test_stop_cancels_background_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = HealthMonitor(_StubStateStore(), _StubEventBus())

    cancel_detected = asyncio.Event()
    sleep_entered = asyncio.Event()

    async def _blocking_sleep(_: float) -> None:
        sleep_entered.set()
        waiter = asyncio.Event()
        try:
            await waiter.wait()
        except asyncio.CancelledError:
            cancel_detected.set()
            raise

    async def _single_check() -> dict[str, object]:
        return {"status": "HEALTHY"}

    async def _noop_store(_: dict[str, object]) -> None:
        return None

    monitor.check_interval = 0.1
    monkeypatch.setattr("src.operational.health_monitor.asyncio.sleep", _blocking_sleep)
    monkeypatch.setattr(monitor, "_perform_health_check", _single_check)
    monkeypatch.setattr(monitor, "_store_health_check", _noop_store)

    await monitor.start()

    await sleep_entered.wait()
    task = monitor._monitor_task
    assert task is not None
    assert not task.done()

    await monitor.stop()

    assert monitor._monitor_task is None
    assert cancel_detected.is_set()
