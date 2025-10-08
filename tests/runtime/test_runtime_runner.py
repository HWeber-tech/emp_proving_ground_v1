from __future__ import annotations

import asyncio

import pytest

from src.runtime.runtime_builder import RuntimeApplication, RuntimeWorkload
from src.runtime.runtime_runner import run_runtime_application
from src.runtime.task_supervisor import TaskSupervisor


@pytest.mark.asyncio
async def test_run_runtime_application_completes_workload() -> None:
    executed: list[str] = []

    async def _workload() -> None:
        executed.append("run")

    runtime_app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="test-ingest",
            description="test workload",
            factory=_workload,
        ),
    )

    await run_runtime_application(runtime_app, register_signals=False)

    assert executed == ["run"]
    assert isinstance(runtime_app.task_supervisor, TaskSupervisor)


class RecordingSupervisor(TaskSupervisor):
    def __init__(self) -> None:
        super().__init__(namespace="test-runtime")
        self.created: list[str | None] = []

    def create(self, coro, *, name: str | None = None, metadata=None):  # type: ignore[override]
        self.created.append(name)
        return super().create(coro, name=name, metadata=metadata)


@pytest.mark.asyncio
async def test_run_runtime_application_reuses_runtime_supervisor() -> None:
    supervisor = RecordingSupervisor()

    async def _workload() -> None:
        await asyncio.sleep(0)

    runtime_app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="test-ingest",
            description="test workload",
            factory=_workload,
        ),
        task_supervisor=supervisor,
    )

    await run_runtime_application(runtime_app, register_signals=False)

    assert runtime_app.task_supervisor is supervisor
    assert "test-ingest-workload" in supervisor.created


@pytest.mark.asyncio
async def test_run_runtime_application_timeout_triggers_shutdown() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    shutdown_called = asyncio.Event()

    async def _workload() -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _shutdown_callback() -> None:
        shutdown_called.set()

    runtime_app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="blocking",
            description="blocks until cancelled",
            factory=_workload,
        ),
    )
    runtime_app.add_shutdown_callback(_shutdown_callback)

    supervisor = TaskSupervisor(namespace="test-runtime-runner")

    await run_runtime_application(
        runtime_app,
        timeout=0.05,
        register_signals=False,
        supervisor=supervisor,
    )

    assert started.is_set()
    assert cancelled.is_set()
    assert shutdown_called.is_set()
    assert supervisor.active_count == 0
    assert runtime_app.task_supervisor is supervisor


@pytest.mark.asyncio
async def test_run_runtime_application_binds_external_supervisor() -> None:
    supervisor = RecordingSupervisor()

    async def _workload() -> None:
        await asyncio.sleep(0)

    runtime_app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="external",
            description="test workload",
            factory=_workload,
        )
    )

    await run_runtime_application(
        runtime_app,
        register_signals=False,
        supervisor=supervisor,
    )

    assert runtime_app.task_supervisor is supervisor
    assert "external-workload" in supervisor.created
