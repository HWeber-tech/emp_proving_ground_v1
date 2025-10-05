import asyncio
import dataclasses
from collections import deque

import pytest

from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    TimescaleBackbonePlan,
)
from src.data_foundation.ingest.production_slice import ProductionIngestSlice
from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import DailyBarIngestPlan
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
)
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings
from src.runtime.task_supervisor import TaskSupervisor


class _DummyEventBus:
    def __init__(self) -> None:
        self.published: deque[tuple[str, dict[str, object]]] = deque()

    def publish_from_sync(self, event) -> None:  # pragma: no cover - defensive path
        self.published.append((event.type, dict(event.payload)))

    def is_running(self) -> bool:
        return True


def _ingest_config(*, schedule: IngestSchedule | None = None) -> InstitutionalIngestConfig:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=5)
    )
    return InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=plan,
        timescale_settings=TimescaleConnectionSettings(url="sqlite:///ingest_test.db"),
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        metadata={},
        schedule=schedule,
    )


@pytest.mark.asyncio
async def test_production_slice_runs_once_and_records_summary() -> None:
    config = _ingest_config()
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-test")

    calls: list[TimescaleBackbonePlan] = []

    class _StubOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            calls.append(plan)
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    orchestrator = _StubOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
    )

    success = await slice_runtime.run_once()

    assert success is True
    assert calls == [config.plan]

    summary = slice_runtime.summary()
    assert summary["last_results"]["daily_bars"]["dimension"] == "daily_bars"
    assert summary["last_error"] is None
    assert summary["last_run_at"] is not None


@pytest.mark.asyncio
async def test_production_slice_supervises_scheduler_tasks() -> None:
    schedule = IngestSchedule(interval_seconds=0.05, jitter_seconds=0.0, max_failures=3)
    config = _ingest_config(schedule=schedule)
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-scheduler")

    run_counts: list[int] = []

    class _CountingOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            run_counts.append(1)
            return {
                "daily_bars": TimescaleIngestResult.empty(
                    dimension="daily_bars", source="yahoo"
                )
            }

    orchestrator = _CountingOrchestrator()

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: orchestrator,
    )

    slice_runtime.start()
    await asyncio.sleep(0.12)
    await slice_runtime.stop()

    assert run_counts  # scheduler executed the ingest callback
    assert supervisor.active_count == 0

    summary = slice_runtime.summary()
    assert summary["services"] is not None
    assert summary["services"]["schedule"]["running"] is False


@pytest.mark.asyncio
async def test_production_slice_handles_disabled_configuration() -> None:
    base = _ingest_config()
    disabled = dataclasses.replace(
        base,
        should_run=False,
        reason="Timescale ingest disabled",
        plan=TimescaleBackbonePlan(),
    )

    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-disabled")

    class _FailingOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            raise AssertionError("Ingest should not execute when disabled")

    slice_runtime = ProductionIngestSlice(
        disabled,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: _FailingOrchestrator(),
    )

    success = await slice_runtime.run_once()

    assert success is False
    slice_runtime.start()  # should no-op
    await slice_runtime.stop()

    summary = slice_runtime.summary()
    assert summary["should_run"] is False
    assert summary["services"] is None
    assert summary["last_results"] == {}
    assert summary["last_error"] == "Timescale ingest disabled"


@pytest.mark.asyncio
async def test_production_slice_records_orchestrator_failures() -> None:
    config = _ingest_config()
    bus = _DummyEventBus()
    supervisor = TaskSupervisor(namespace="ingest-error")

    class _FailingOrchestrator:
        def run(self, *, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            _ = plan  # defensive: plan is not needed in this stub
            raise RuntimeError("Timescale ingest execution failed")

    slice_runtime = ProductionIngestSlice(
        config,
        bus,
        supervisor,
        orchestrator_factory=lambda settings, publisher: _FailingOrchestrator(),
    )

    success = await slice_runtime.run_once()

    assert success is False

    summary = slice_runtime.summary()
    assert summary["last_results"] == {}
    assert summary["last_run_at"] is None
    assert summary["last_error"] == "Timescale ingest execution failed"

    await slice_runtime.stop()
