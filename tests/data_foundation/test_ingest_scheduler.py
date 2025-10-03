import asyncio
from datetime import UTC, datetime, timedelta

import pytest


pytestmark = [pytest.mark.guardrail]

from src.core.event_bus import AsyncEventBus
from src.data_foundation.ingest.scheduler import (
    IngestSchedule,
    IngestSchedulerState,
    TimescaleIngestScheduler,
)
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerStatus,
    build_scheduler_snapshot,
    format_scheduler_markdown,
    publish_scheduler_snapshot,
)
from src.runtime.task_supervisor import TaskSupervisor


@pytest.mark.asyncio()
async def test_scheduler_runs_until_stopped() -> None:
    runs: list[int] = []
    trigger = asyncio.Event()

    async def _callback() -> bool:
        runs.append(len(runs) + 1)
        if len(runs) >= 3:
            trigger.set()
        return True

    scheduler = TimescaleIngestScheduler(
        schedule=IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=5),
        run_callback=_callback,
    )
    scheduler.start()

    await trigger.wait()
    await scheduler.stop()

    assert len(runs) >= 3
    assert scheduler.running is False
    state = scheduler.state()
    assert isinstance(state, IngestSchedulerState)
    assert state.running is False
    assert state.last_started_at is not None


@pytest.mark.asyncio()
async def test_scheduler_stops_after_max_failures() -> None:
    runs = 0

    async def _callback() -> bool:
        nonlocal runs
        runs += 1
        return False

    scheduler = TimescaleIngestScheduler(
        schedule=IngestSchedule(interval_seconds=0.02, jitter_seconds=0.0, max_failures=2),
        run_callback=_callback,
    )
    task = scheduler.start()

    await asyncio.sleep(0.15)
    await task

    assert runs >= 2
    assert scheduler.running is False
    state = scheduler.state()
    assert state.consecutive_failures >= 2
    assert state.next_run_at is None


@pytest.mark.asyncio()
async def test_scheduler_invokes_jitter(monkeypatch) -> None:
    jitter_calls: list[tuple[float, float]] = []
    done = asyncio.Event()
    runs = 0

    def _fake_uniform(low: float, high: float) -> float:
        jitter_calls.append((low, high))
        return 0.0

    monkeypatch.setattr("src.data_foundation.ingest.scheduler.random.uniform", _fake_uniform)

    async def _callback() -> bool:
        nonlocal runs
        runs += 1
        if runs >= 2:
            done.set()
        return True

    scheduler = TimescaleIngestScheduler(
        schedule=IngestSchedule(interval_seconds=0.05, jitter_seconds=0.02, max_failures=3),
        run_callback=_callback,
    )
    scheduler.start()

    await done.wait()
    await scheduler.stop()

    assert runs >= 2
    assert jitter_calls
    assert all(call == (-0.02, 0.02) for call in jitter_calls)


@pytest.mark.asyncio()
async def test_scheduler_tracks_task_with_supervisor() -> None:
    supervisor = TaskSupervisor(namespace="test-ingest")
    started = asyncio.Event()

    async def _callback() -> bool:
        started.set()
        return True

    scheduler = TimescaleIngestScheduler(
        schedule=IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3),
        run_callback=_callback,
        task_supervisor=supervisor,
        task_metadata={"pipeline": "institutional"},
        task_name="supervised-timescale-ingest",
    )

    scheduler.start()
    await started.wait()
    await asyncio.sleep(0)

    snapshots = supervisor.describe()
    assert snapshots, "Supervisor should expose active ingest task telemetry"
    supervised = next(
        snapshot for snapshot in snapshots if snapshot["name"] == "supervised-timescale-ingest"
    )
    assert supervised["metadata"]["component"] == "timescale_ingest.scheduler"
    assert supervised["metadata"]["pipeline"] == "institutional"

    await scheduler.stop()
    await asyncio.sleep(0)
    assert supervisor.active_count == 0


@pytest.mark.asyncio()
async def test_scheduler_state_tracks_next_run(monkeypatch) -> None:
    # Freeze random.uniform to simplify assertions.
    monkeypatch.setattr(
        "src.data_foundation.ingest.scheduler.random.uniform",
        lambda low, high: 0.0,
    )

    started = asyncio.Event()

    async def _callback() -> bool:
        started.set()
        await asyncio.sleep(0)
        return True

    scheduler = TimescaleIngestScheduler(
        schedule=IngestSchedule(interval_seconds=0.05, jitter_seconds=0.01, max_failures=3),
        run_callback=_callback,
    )
    scheduler.start()

    await started.wait()
    await asyncio.sleep(0.06)
    state = scheduler.state()
    assert state.running is True
    assert state.last_started_at is not None
    assert state.last_completed_at is not None
    assert state.last_success_at is not None
    assert state.next_run_at is not None
    assert state.next_run_at >= state.last_completed_at
    await scheduler.stop()
    final_state = scheduler.state()
    assert final_state.running is False
    assert final_state.next_run_at is None


def test_build_scheduler_snapshot_disabled() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = build_scheduler_snapshot(
        enabled=False,
        schedule=None,
        state=None,
        now=now,
    )
    assert snapshot.status is IngestSchedulerStatus.warn
    assert snapshot.enabled is False
    assert snapshot.running is False
    assert any("disabled" in issue.lower() for issue in snapshot.issues)


def test_build_scheduler_snapshot_running_ok() -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    schedule = IngestSchedule(interval_seconds=120.0, jitter_seconds=10.0, max_failures=5)
    state = IngestSchedulerState(
        running=True,
        last_started_at=now - timedelta(seconds=5),
        last_completed_at=now - timedelta(seconds=4),
        last_success_at=now - timedelta(seconds=4),
        consecutive_failures=0,
        next_run_at=now + timedelta(seconds=120),
        interval_seconds=120.0,
        jitter_seconds=10.0,
        max_failures=5,
    )
    snapshot = build_scheduler_snapshot(
        enabled=True,
        schedule=schedule,
        state=state,
        now=now,
    )
    assert snapshot.status is IngestSchedulerStatus.ok
    assert snapshot.running is True
    assert snapshot.issues == tuple()
    markdown = format_scheduler_markdown(snapshot)
    assert "Status" in markdown


def test_build_scheduler_snapshot_flags_staleness() -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    schedule = IngestSchedule(interval_seconds=60.0, jitter_seconds=0.0, max_failures=3)
    state = IngestSchedulerState(
        running=True,
        last_started_at=now - timedelta(minutes=10),
        last_completed_at=now - timedelta(minutes=10),
        last_success_at=now - timedelta(minutes=10),
        consecutive_failures=1,
        next_run_at=now - timedelta(minutes=1),
        interval_seconds=60.0,
        jitter_seconds=0.0,
        max_failures=3,
    )
    snapshot = build_scheduler_snapshot(
        enabled=True,
        schedule=schedule,
        state=state,
        now=now,
    )
    assert snapshot.status is not IngestSchedulerStatus.ok
    assert any("overdue" in issue.lower() for issue in snapshot.issues)


@pytest.mark.asyncio()
async def test_publish_scheduler_snapshot_emits_event() -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    schedule = IngestSchedule(interval_seconds=30.0, jitter_seconds=0.0, max_failures=2)
    state = IngestSchedulerState(
        running=True,
        last_started_at=now - timedelta(seconds=5),
        last_completed_at=now - timedelta(seconds=4),
        last_success_at=now - timedelta(seconds=4),
        consecutive_failures=0,
        next_run_at=now + timedelta(seconds=30),
        interval_seconds=30.0,
        jitter_seconds=0.0,
        max_failures=2,
    )
    snapshot = build_scheduler_snapshot(
        enabled=True,
        schedule=schedule,
        state=state,
        now=now,
    )

    bus = AsyncEventBus()
    received: list[object] = []

    async def _handler(event) -> None:
        received.append(event)

    bus.subscribe("telemetry.ingest.scheduler", _handler)
    await bus.start()
    try:
        await publish_scheduler_snapshot(bus, snapshot)
        await asyncio.sleep(0)
    finally:
        await bus.stop()

    assert received
    event = received[0]
    assert event.type == "telemetry.ingest.scheduler"
    assert event.payload["status"] == snapshot.status.value
    assert "markdown" in event.payload


def test_ingest_schedule_validation_guards_negative_values() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        IngestSchedule(interval_seconds=0)
    with pytest.raises(ValueError, match="jitter_seconds"):
        IngestSchedule(interval_seconds=1, jitter_seconds=-0.1)
    with pytest.raises(ValueError, match="max_failures"):
        IngestSchedule(interval_seconds=1, jitter_seconds=0, max_failures=-1)


def test_ingest_scheduler_state_serialises_to_json_primitives() -> None:
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    state = IngestSchedulerState(
        running=True,
        last_started_at=now,
        last_completed_at=now,
        last_success_at=now,
        consecutive_failures=2,
        next_run_at=now + timedelta(seconds=30),
        interval_seconds=30.0,
        jitter_seconds=5.0,
        max_failures=4,
    )

    snapshot = state.as_dict()

    assert snapshot["running"] is True
    assert snapshot["last_started_at"].endswith("+00:00")
    assert snapshot["consecutive_failures"] == 2
    assert snapshot["interval_seconds"] == 30.0
    assert snapshot["max_failures"] == 4
