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
async def test_supervisor_snapshot_reports_age_in_seconds() -> None:
    supervisor = TaskSupervisor(namespace="age-test")
    release = asyncio.Event()

    async def _await_release() -> None:
        await release.wait()

    supervisor.create(_await_release(), name="age-task")
    await asyncio.sleep(0.02)

    first_snapshot = supervisor.describe()[0]
    assert "age_seconds" in first_snapshot
    initial_age = first_snapshot["age_seconds"]
    assert isinstance(initial_age, float)
    assert initial_age >= 0.0

    await asyncio.sleep(0.02)
    second_snapshot = supervisor.describe()[0]
    assert second_snapshot["age_seconds"] >= initial_age

    release.set()
    await supervisor.cancel_all()


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
async def test_supervisor_auto_names_tasks_without_name() -> None:
    supervisor = TaskSupervisor(namespace="auto")
    started = asyncio.Event()
    release = asyncio.Event()

    async def _workload() -> None:
        started.set()
        await release.wait()

    task = supervisor.create(_workload())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    snapshots = supervisor.describe()
    assert snapshots, "expected auto-generated task snapshot"
    auto_name = snapshots[0]["name"]
    assert isinstance(auto_name, str) and auto_name.startswith("auto-task-")

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_assigns_name_when_tracking_default_named_task() -> None:
    supervisor = TaskSupervisor(namespace="auto-track")
    release = asyncio.Event()

    async def _external() -> None:
        await release.wait()

    task = asyncio.create_task(_external())  # default event-loop naming (Task-*)
    supervisor.track(task)
    await asyncio.sleep(0)

    snapshots = supervisor.describe()
    assert snapshots
    generated_name = snapshots[0]["name"]
    assert isinstance(generated_name, str) and generated_name.startswith("auto-track-task-")

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await supervisor.cancel_all()


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


@pytest.mark.asyncio()
async def test_supervisor_hang_timeout_triggers_restart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = TaskSupervisor(namespace="test-hang")
    attempts = 0
    completion_flag = asyncio.Event()

    async def _workload() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(0.2)
        else:
            completion_flag.set()

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        task = supervisor.create(
            _workload(),
            name="hang-task",
            restart_callback=_workload,
            max_restarts=3,
            restart_backoff=0.0,
            hang_timeout=0.05,
        )
        snapshots = supervisor.describe()
        assert snapshots and pytest.approx(0.05, rel=1e-3) == snapshots[0]["hang_timeout_seconds"]
        await asyncio.wait_for(completion_flag.wait(), timeout=1.0)
        await asyncio.wait_for(task, timeout=1.0)

    assert attempts == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("hang-task exceeded hang timeout" in message for message in messages)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_loop_task_factory_tracks_asyncio_create_task() -> None:
    supervisor = TaskSupervisor(namespace="loop-factory")
    loop = asyncio.get_running_loop()
    supervisor.install_loop_task_factory(loop=loop, metadata={"origin": "loop"})

    ready = asyncio.Event()
    release = asyncio.Event()

    async def _worker() -> None:
        ready.set()
        await release.wait()

    task = asyncio.create_task(_worker(), name="loop-worker")

    await asyncio.wait_for(ready.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert supervisor.is_tracked(task)
    snapshots = supervisor.describe()
    assert any(
        (snap.get("metadata") or {}).get("origin") == "loop"
        for snap in snapshots
    )

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await supervisor.cancel_all()
    assert loop.get_task_factory() is None


@pytest.mark.asyncio()
async def test_supervisor_loop_task_factory_reinstall_updates_metadata() -> None:
    supervisor = TaskSupervisor(namespace="loop-factory-reinstall")
    loop = asyncio.get_running_loop()
    supervisor.install_loop_task_factory(loop=loop, metadata={"origin": "first"})
    supervisor.install_loop_task_factory(loop=loop, metadata={"origin": "second"})

    triggered = asyncio.Event()
    release = asyncio.Event()

    async def _job() -> None:
        triggered.set()
        await release.wait()

    task = asyncio.create_task(_job(), name="loop-reinstall-worker")

    await asyncio.wait_for(triggered.wait(), timeout=1.0)
    await asyncio.sleep(0)

    snapshots = supervisor.describe()
    assert any(
        (snap.get("metadata") or {}).get("origin") == "second"
        for snap in snapshots
    )

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_hang_timeout_without_restart(caplog: pytest.LogCaptureFixture) -> None:
    supervisor = TaskSupervisor(namespace="test-hang-fail")

    async def _hang() -> None:
        await asyncio.sleep(0.2)

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        task = supervisor.create(
            _hang(),
            name="hang-failure",
            hang_timeout=0.05,
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=1.0)

    messages = [record.getMessage() for record in caplog.records]
    assert any("hang-failure exceeded hang timeout" in message for message in messages)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_supervisor_cancel_all_reports_hung_tasks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = TaskSupervisor(namespace="test-hung", cancel_timeout=0.05)
    release_event = asyncio.Event()

    async def _hung() -> None:
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            await release_event.wait()
            raise

    hung_task = supervisor.create(
        _hung(), name="hung-task", metadata={"component": "drift-monitor"}
    )

    await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        await supervisor.cancel_all()

    messages = [record.getMessage() for record in caplog.records]
    assert any("hung-task" in message for message in messages)
    assert any("component" in message for message in messages)

    release_event.set()
    try:
        await hung_task
    except asyncio.CancelledError:
        pass
