"""Async scheduler for recurring Timescale ingest runs with failure guardrails."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Coroutine, Protocol


logger = logging.getLogger(__name__)


RunCallback = Callable[[], Awaitable[bool | None]]


class _TaskFactory(Protocol):
    """Protocol describing the subset of :func:`asyncio.create_task` we rely on."""

    def __call__(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        ...


@dataclass(frozen=True)
class IngestSchedulerState:
    """Snapshot describing the most recent scheduler activity."""

    running: bool
    last_started_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int = 0
    next_run_at: datetime | None = None
    interval_seconds: float = 0.0
    jitter_seconds: float = 0.0
    max_failures: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialise the scheduler state into JSON-friendly primitives."""

        def _iso(moment: datetime | None) -> str | None:
            return moment.isoformat() if moment is not None else None

        return {
            "running": self.running,
            "last_started_at": _iso(self.last_started_at),
            "last_completed_at": _iso(self.last_completed_at),
            "last_success_at": _iso(self.last_success_at),
            "consecutive_failures": self.consecutive_failures,
            "next_run_at": _iso(self.next_run_at),
            "interval_seconds": self.interval_seconds,
            "jitter_seconds": self.jitter_seconds,
            "max_failures": self.max_failures,
        }


@dataclass(frozen=True)
class IngestSchedule:
    """Configuration describing how frequently to execute Timescale ingest."""

    interval_seconds: float
    jitter_seconds: float = 0.0
    max_failures: int = 3

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds cannot be negative")
        if self.max_failures < 0:
            raise ValueError("max_failures cannot be negative")


class TimescaleIngestScheduler:
    """Run Timescale ingest on an interval until stopped or too many failures occur."""

    def __init__(
        self,
        *,
        schedule: IngestSchedule,
        run_callback: RunCallback,
        task_name: str = "timescale-ingest-scheduler",
        task_factory: _TaskFactory | None = None,
    ) -> None:
        self._schedule = schedule
        self._run_callback = run_callback
        self._task_name = task_name
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._failure_count = 0
        self._logger = logging.getLogger(f"{__name__}.TimescaleIngestScheduler")
        self._last_started_at: datetime | None = None
        self._last_completed_at: datetime | None = None
        self._last_success_at: datetime | None = None
        self._next_run_at: datetime | None = None
        self._task_factory = task_factory

    @property
    def running(self) -> bool:
        """Return True when the scheduler loop is active."""

        return self._task is not None and not self._task.done()

    def start(
        self,
        *,
        task_factory: _TaskFactory | None = None,
    ) -> asyncio.Task[None]:
        """Start the scheduler loop and return the background task."""

        if self.running:
            return self._task  # type: ignore[return-value]

        self._stop_event = asyncio.Event()
        self._failure_count = 0
        self._next_run_at = None
        factory = task_factory or self._task_factory
        if factory is not None:
            task = factory(self._run_loop(), name=self._task_name)
            if not isinstance(task, asyncio.Task):
                raise TypeError("Task factory must return an asyncio.Task")
            self._task = task
        else:
            self._task = asyncio.create_task(self._run_loop(), name=self._task_name)
        return self._task

    async def stop(self) -> None:
        """Signal the scheduler to stop and wait for the task to exit."""

        if not self.running:
            return

        assert self._task is not None  # typing guard
        assert self._stop_event is not None

        self._stop_event.set()
        task = self._task
        self._task = None
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - defensive
            pass

    async def _run_loop(self) -> None:
        assert self._stop_event is not None

        while True:
            if self._stop_event.is_set():
                break

            try:
                self._last_started_at = datetime.now(tz=UTC)
                outcome = await self._run_callback()
            except asyncio.CancelledError:  # pragma: no cover - passthrough cancellation
                raise
            except Exception:
                self._logger.exception("Timescale ingest run raised an exception")
                success = False
            else:
                success = True if outcome is None else bool(outcome)
            finally:
                self._last_completed_at = datetime.now(tz=UTC)

            if success:
                self._failure_count = 0
                self._last_success_at = self._last_completed_at
            else:
                self._failure_count += 1
                max_failures = self._schedule.max_failures
                if max_failures and self._failure_count >= max_failures:
                    self._logger.error(
                        "Timescale ingest scheduler stopping after %s consecutive failures",
                        self._failure_count,
                    )
                    break

            delay = self._schedule.interval_seconds
            jitter = self._schedule.jitter_seconds
            if jitter:
                try:
                    delay += random.uniform(-jitter, jitter)
                except Exception:  # pragma: no cover - random failure improbable
                    self._logger.debug(
                        "random.uniform failed; continuing without jitter", exc_info=True
                    )
                delay = max(1.0, delay)

            completed_at = self._last_completed_at or datetime.now(tz=UTC)
            self._next_run_at = completed_at + timedelta(seconds=delay)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
            else:
                self._next_run_at = None

        self._logger.info("Timescale ingest scheduler loop exited")
        self._next_run_at = None

    def state(self) -> IngestSchedulerState:
        """Return the latest scheduler telemetry."""

        return IngestSchedulerState(
            running=self.running,
            last_started_at=self._last_started_at,
            last_completed_at=self._last_completed_at,
            last_success_at=self._last_success_at,
            consecutive_failures=self._failure_count,
            next_run_at=self._next_run_at,
            interval_seconds=self._schedule.interval_seconds,
            jitter_seconds=self._schedule.jitter_seconds,
            max_failures=self._schedule.max_failures,
        )


__all__ = ["IngestSchedule", "IngestSchedulerState", "TimescaleIngestScheduler"]
