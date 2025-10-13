"""Service wrapper for the operational data backbone pipeline.

The roadmap expects the institutional data backbone (Timescale → Redis →
Kafka → sensory) to run as a managed service with supervised background
tasks, repeatable ingest drills, and easy access to readiness evidence.  The
existing building blocks – :class:`RealDataManager` and
:class:`OperationalBackbonePipeline` – provide the core functionality, but
runtime and operations tooling benefit from a higher-level facade that keeps
the moving pieces coordinated.

``OperationalBackboneService`` packages the manager, pipeline, streaming
bridge, and optional ingest scheduler into a single lifecycle object.  It
offers a terse API for:

* Executing ingest slices (store → cache → stream) and capturing the
  resulting evidence.
* Starting/stopping the Kafka streaming bridge while tracking the background
  task via :class:`~src.runtime.task_supervisor.TaskSupervisor` metadata.
* Launching the recurring ingest scheduler so live-shadow runs mirror
  production cadence.
* Summarising cache metrics and connector health for readiness dashboards.

The service deliberately avoids hiding the underlying components – callers
can still access the manager or pipeline when bespoke hooks are required –
but it centralises lifecycle management and clean shutdown semantics so the
rest of the runtime does not need to duplicate bookkeeping logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data_foundation.ingest.scheduler import IngestSchedule, TimescaleIngestScheduler
from src.data_foundation.ingest.timescale_pipeline import TimescaleBackbonePlan
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalBackboneResult,
    OperationalIngestRequest,
)
from src.data_integration.real_data_integration import BackboneConnectivityReport, RealDataManager
from src.runtime.task_supervisor import TaskSupervisor


FetchDailyFn = Callable[[list[str], int], pd.DataFrame]
FetchIntradayFn = Callable[[list[str], int, str], pd.DataFrame]
FetchMacroFn = Callable[[str, str], Sequence[Mapping[str, object]]]


def _task_active(task: asyncio.Task[Any] | None) -> bool:
    return task is not None and not task.done()


@dataclass(slots=True)
class OperationalBackboneService:
    """Coordinate the operational data backbone components as a lifecycle service."""

    manager: RealDataManager
    pipeline: OperationalBackbonePipeline
    owns_manager: bool = False
    owns_pipeline: bool = False
    _scheduler: TimescaleIngestScheduler | None = field(init=False, default=None)
    _streaming_task: asyncio.Task[None] | None = field(init=False, default=None)
    _pipeline_closes_manager: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        self._pipeline_closes_manager = bool(
            getattr(self.pipeline, "_shutdown_manager_on_close", True)
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OperationalBackboneService":  # pragma: no cover - sugar
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - sugar
        await self.shutdown()

    async def shutdown(self) -> None:
        """Stop streaming/scheduler tasks and dispose owned resources."""

        await self.stop_scheduler()
        await self.stop_streaming()

        if self.owns_pipeline:
            await self.pipeline.shutdown()
        else:  # ensure the pipeline releases transient resources
            await self.pipeline.stop_streaming()

        manager_closed = False
        if not self._pipeline_closes_manager and self.owns_manager:
            await self.manager.shutdown()
            manager_closed = True
        if self._pipeline_closes_manager and self.owns_manager and not manager_closed:
            await self.manager.shutdown()

    # ------------------------------------------------------------------
    # Ingest + streaming orchestration
    # ------------------------------------------------------------------

    async def ingest_once(
        self,
        request: OperationalIngestRequest,
        *,
        fetch_daily: FetchDailyFn | None = None,
        fetch_intraday: FetchIntradayFn | None = None,
        fetch_macro: FetchMacroFn | None = None,
        poll_consumer: bool = True,
    ) -> OperationalBackboneResult:
        """Execute a single ingest slice and return the pipeline result."""

        return await self.pipeline.execute(
            request,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            fetch_macro=fetch_macro,
            poll_consumer=poll_consumer,
        )

    async def ensure_streaming(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
        task_name: str | None = None,
    ) -> asyncio.Task[None] | None:
        """Launch the Kafka streaming bridge if it is not already active."""

        if _task_active(self._streaming_task):
            return self._streaming_task

        task = await self.pipeline.start_streaming(task_name=task_name, metadata=metadata)
        self._streaming_task = task
        return task

    async def stop_streaming(self) -> None:
        """Stop the streaming bridge if it is running."""

        if not _task_active(self._streaming_task):
            self._streaming_task = None
            await self.pipeline.stop_streaming()
            return

        await self.pipeline.stop_streaming()
        self._streaming_task = None

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    async def start_scheduler(
        self,
        plan_factory: Callable[[], TimescaleBackbonePlan],
        schedule: IngestSchedule,
        *,
        fetch_daily: FetchDailyFn | None = None,
        fetch_intraday: FetchIntradayFn | None = None,
        fetch_macro: FetchMacroFn | None = None,
        metadata: Mapping[str, object] | None = None,
        task_supervisor: TaskSupervisor | None = None,
    ) -> TimescaleIngestScheduler:
        """Launch the recurring ingest scheduler for the managed manager."""

        if self._scheduler is not None and self._scheduler.running:
            return self._scheduler

        scheduler = self.manager.start_ingest_scheduler(
            plan_factory,
            schedule,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            fetch_macro=fetch_macro,
            metadata=metadata,
            task_supervisor=task_supervisor,
        )
        self._scheduler = scheduler
        return scheduler

    async def stop_scheduler(self) -> None:
        """Stop the ingest scheduler if it is active."""

        if self._scheduler is None:
            return
        await self.manager.stop_ingest_scheduler()
        self._scheduler = None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def streaming_active(self) -> bool:
        return _task_active(self._streaming_task)

    @property
    def scheduler_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    @property
    def task_supervisor(self) -> TaskSupervisor | None:
        return getattr(self.pipeline, "_task_supervisor", None)

    def task_snapshots(self) -> tuple[Mapping[str, object], ...]:
        supervisor = self.task_supervisor
        if supervisor is None:
            return tuple()
        try:
            return tuple(supervisor.describe())
        except Exception:
            return tuple()

    def streaming_snapshots(self) -> Mapping[str, Mapping[str, Any]]:
        return getattr(self.pipeline, "streaming_snapshots", {})

    def cache_metrics(self, *, reset: bool = False) -> Mapping[str, Any]:
        return dict(self.manager.cache_metrics(reset=reset))

    def connectivity_report(self) -> BackboneConnectivityReport:
        return self.manager.connectivity_report()

    def scheduler_state(self) -> Mapping[str, object] | None:
        if self._scheduler is None:
            return None
        try:
            return self._scheduler.state().as_dict()
        except Exception:
            return None

    def summary(self) -> dict[str, Any]:  # pragma: no cover - convenience wrapper
        payload: dict[str, Any] = {
            "streaming": self.streaming_active,
            "scheduler_running": self.scheduler_running,
            "cache_metrics": self.cache_metrics(reset=False),
        }
        report = self.connectivity_report()
        payload["connectivity"] = report.as_dict()
        scheduler_state = self.scheduler_state()
        if scheduler_state is not None:
            payload["scheduler_state"] = scheduler_state
        snapshots = self.streaming_snapshots()
        if snapshots:
            payload["streaming_snapshots"] = dict(snapshots)
        tasks = self.task_snapshots()
        if tasks:
            payload["task_snapshots"] = list(tasks)
        return payload


__all__ = ["OperationalBackboneService"]
