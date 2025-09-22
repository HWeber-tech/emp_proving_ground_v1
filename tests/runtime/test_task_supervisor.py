import asyncio
import logging

import pytest

from src.runtime.task_supervisor import TaskSupervisor


@pytest.mark.asyncio()
async def test_supervisor_tracks_created_tasks_and_cancels() -> None:
    supervisor = TaskSupervisor(namespace="test")

    async def _worker() -> None:
        await asyncio.sleep(0.05)

    supervisor.create(_worker(), name="test-worker")
    assert supervisor.active_count == 1
    assert any(detail["name"] == "test-worker" for detail in supervisor.describe())

    await supervisor.cancel_all()
    assert supervisor.active_count == 0


@pytest.mark.asyncio()
async def test_supervisor_logs_failures(caplog: pytest.LogCaptureFixture) -> None:
    supervisor = TaskSupervisor(namespace="test", cancel_timeout=0.1)

    async def _failing() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        supervisor.create(_failing(), name="boom-task")
        for _ in range(5):
            await asyncio.sleep(0)
        assert any("boom-task" in record.getMessage() for record in caplog.records), caplog.records

    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_tracks_external_tasks_and_metadata() -> None:
    supervisor = TaskSupervisor(namespace="test")
    stop_event = asyncio.Event()

    async def _external() -> None:
        await stop_event.wait()

    task = asyncio.create_task(_external(), name="external-task")
    supervisor.track(task, metadata={"component": "kafka"})

    details = supervisor.describe()
    assert details and details[0]["metadata"] == {"component": "kafka"}

    stop_event.set()
    await asyncio.sleep(0)  # let the task exit cleanly
    await supervisor.cancel_all()
    assert supervisor.active_count == 0
