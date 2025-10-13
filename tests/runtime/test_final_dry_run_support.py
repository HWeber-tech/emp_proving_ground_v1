from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Coroutine

import pytest

from src.core.event_bus import Event, SubscriptionHandle
from src.runtime.final_dry_run_support import (
    FinalDryRunPerformanceWriter,
    configure_final_dry_run_support,
)


class _StubEventBus:
    def __init__(self) -> None:
        self._registrations: list[tuple[SubscriptionHandle, Callable[[Event], Any]]] = []

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Any],
    ) -> SubscriptionHandle:
        handle = SubscriptionHandle(
            id=len(self._registrations) + 1,
            event_type=event_type,
            handler=handler,
        )
        self._registrations.append((handle, handler))
        return handle

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        self._registrations = [
            (registered, handler)
            for registered, handler in self._registrations
            if registered.id != handle.id
        ]

    def get_handlers(self, event_type: str) -> list[Callable[[Event], Any]]:
        return [
            handler
            for handle, handler in self._registrations
            if handle.event_type == event_type
        ]

    @property
    def registrations(self) -> list[tuple[SubscriptionHandle, Callable[[Event], Any]]]:
        return list(self._registrations)


class _StubApp:
    def __init__(self, *, extras: dict[str, str], base_path: Path) -> None:
        self.config = SimpleNamespace(extras=dict(extras))
        self.event_bus = _StubEventBus()
        self._tasks: list[asyncio.Task[Any]] = []
        self._cleanup_callbacks: list[Callable[[], Any]] = []
        self.base_path = base_path

    def create_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task

    def add_cleanup_callback(self, callback: Callable[[], Any]) -> None:
        self._cleanup_callbacks.append(callback)

    async def run_cleanup(self) -> None:
        for callback in list(self._cleanup_callbacks):
            result = callback()
            if asyncio.iscoroutine(result):
                await result
        for task in list(self._tasks):
            if task.done():
                continue
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_performance_writer_persists_snapshot(tmp_path: Path) -> None:
    app = _StubApp(extras={}, base_path=tmp_path)
    writer = FinalDryRunPerformanceWriter(tmp_path / "performance.json", run_label="Phase II UAT")
    writer.install(app)

    placeholder = json.loads((tmp_path / "performance.json").read_text(encoding="utf-8"))
    assert placeholder["status"] == "waiting"
    assert placeholder["run_label"] == "Phase II UAT"

    handler = app.event_bus.get_handlers("telemetry.strategy.performance")[0]
    event = Event(
        type="telemetry.strategy.performance",
        payload={
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "status": "pass",
        },
        source="unit-test",
    )

    await handler(event)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    payload = json.loads((tmp_path / "performance.json").read_text(encoding="utf-8"))
    assert payload["sequence"] == 1
    assert payload["snapshot"]["status"] == "pass"
    assert payload["run_label"] == "Phase II UAT"
    assert payload["event_source"] == "unit-test"

    await writer.close()
    await app.run_cleanup()
    assert not app.event_bus.get_handlers("telemetry.strategy.performance")


@pytest.mark.asyncio
async def test_configure_final_dry_run_support_installs_writer(tmp_path: Path) -> None:
    extras = {
        "FINAL_DRY_RUN_PERFORMANCE_PATH": str(tmp_path / "perf.json"),
        "FINAL_DRY_RUN_LABEL": "Dry Run",
    }
    app = _StubApp(extras=extras, base_path=tmp_path)

    configure_final_dry_run_support(app)

    handlers = app.event_bus.get_handlers("telemetry.strategy.performance")
    assert handlers, "Expected writer subscription to be registered"
    assert (tmp_path / "perf.json").exists()

    await app.run_cleanup()


def test_configure_final_dry_run_support_skips_when_missing_path(tmp_path: Path) -> None:
    app = _StubApp(extras={}, base_path=tmp_path)

    configure_final_dry_run_support(app)

    assert not app.event_bus.registrations
    assert not list(tmp_path.glob("*.json")), "No evidence files should be created"
