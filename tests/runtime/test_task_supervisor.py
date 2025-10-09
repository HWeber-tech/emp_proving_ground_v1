import asyncio
import logging

import pytest

from src.runtime.task_supervisor import TaskSupervisor


@pytest.mark.asyncio()
async def test_supervisor_tracks_created_tasks_and_cancels() -> None:
    supervisor = TaskSupervisor(namespace="test")
    assert supervisor.namespace == "test"

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


@pytest.mark.asyncio()
async def test_supervisor_failure_does_not_cancel_other_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = TaskSupervisor(namespace="test", cancel_timeout=0.05)
    running_flag = asyncio.Event()
    release_event = asyncio.Event()

    async def _failing_ingest() -> None:
        await running_flag.wait()
        raise RuntimeError("ingest burst")

    async def _drift_monitor() -> None:
        running_flag.set()
        await release_event.wait()

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        failing_task = supervisor.create(
            _failing_ingest(),
            name="timescale-ingest-runner",
        )
        drift_task = supervisor.create(
            _drift_monitor(),
            name="drift-monitor",
        )

        # allow the failing task to complete and be logged
        while not failing_task.done():
            await asyncio.sleep(0)

        await asyncio.sleep(0)
        assert caplog.records, "expected supervisor to log the ingest failure"
        assert any(
            "timescale-ingest-runner" in record.getMessage()
            for record in caplog.records
        )

    assert not drift_task.done(), "supervisor cancelled an unrelated task"

    release_event.set()
    await asyncio.sleep(0)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_restarts_task_until_success(caplog: pytest.LogCaptureFixture) -> None:
    supervisor = TaskSupervisor(namespace="test-restart", cancel_timeout=0.1)
    attempts = 0
    completion_flag = asyncio.Event()
    release_event = asyncio.Event()

    async def _worker() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"attempt-{attempts}")
        completion_flag.set()
        await release_event.wait()

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        task = supervisor.create(
            _worker(),
            name="restartable-task",
            restart_callback=_worker,
            max_restarts=5,
            restart_backoff=0.0,
        )
        await asyncio.wait_for(completion_flag.wait(), timeout=1.0)
        snapshot = supervisor.describe()[0]
        assert snapshot.get("restarts") == 2
        release_event.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert attempts == 3
    messages = [record.getMessage() for record in caplog.records]
    assert any("restartable-task" in message for message in messages)
    await supervisor.cancel_all()
