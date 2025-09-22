"""Async task supervision utilities for runtime background workloads."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Mapping, MutableMapping, Sequence

__all__ = ["TaskSupervisor", "TaskSnapshot"]


@dataclass(slots=True, frozen=True)
class TaskSnapshot:
    """Snapshot describing a tracked task for diagnostics and summaries."""

    name: str | None
    state: str
    created_at: datetime
    metadata: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class _TrackedTask:
    """Internal record describing an active task."""

    name: str | None
    created_at: datetime
    metadata: Mapping[str, Any] | None


class TaskSupervisor:
    """Track background asyncio tasks and coordinate graceful shutdown."""

    def __init__(
        self,
        *,
        namespace: str = "runtime",
        logger: logging.Logger | None = None,
        cancel_timeout: float = 5.0,
    ) -> None:
        self._namespace = namespace
        self._logger = logger or logging.getLogger(f"{__name__}.{namespace}")
        self._cancel_timeout = cancel_timeout
        self._tasks: MutableMapping[asyncio.Task[Any], _TrackedTask] = {}
        self._closing: bool = False

    @property
    def active_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        """Return the currently tracked tasks that have not finished."""

        return tuple(self._tasks.keys())

    @property
    def active_count(self) -> int:
        """Return the number of active tracked tasks."""

        return len(self._tasks)

    def create(
        self,
        coro: Awaitable[Any],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        """Create and track a background task."""

        if self._closing:
            raise RuntimeError("TaskSupervisor is shutting down; cannot create new tasks")

        task = asyncio.create_task(coro, name=name)
        self._register(task, metadata)
        return task

    def track(
        self,
        task: asyncio.Task[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        """Track an externally created task."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("TaskSupervisor.track expects an asyncio.Task instance")
        if self._closing:
            raise RuntimeError("TaskSupervisor is shutting down; cannot track new tasks")

        self._register(task, metadata)
        return task

    def describe(self) -> list[dict[str, Any]]:
        """Return lightweight metadata for active tasks."""

        snapshots: list[dict[str, Any]] = []
        for task, record in self._tasks.items():
            snapshot = TaskSnapshot(
                name=record.name,
                state=self._task_state(task),
                created_at=record.created_at,
                metadata=record.metadata,
            )
            snapshots.append(snapshot.as_dict())
        return snapshots

    async def cancel_all(self) -> None:
        """Cancel tracked tasks and wait for them to exit."""

        if self._closing:
            return

        tasks: Sequence[asyncio.Task[Any]] = tuple(self._tasks.keys())
        if not tasks:
            return

        self._closing = True
        for task in tasks:
            if not task.done():
                try:
                    task.cancel()
                except Exception:  # pragma: no cover - defensive
                    self._logger.exception("Failed to cancel task %s", self._task_name(task))

        async def _await_tasks() -> Sequence[Any]:
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await asyncio.wait_for(_await_tasks(), timeout=self._cancel_timeout)
        except asyncio.TimeoutError:
            self._logger.error(
                "Timeout while awaiting %s background tasks during shutdown", len(tasks)
            )
        finally:
            self._closing = False
            self._tasks.clear()

    def _register(
        self,
        task: asyncio.Task[Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        record = _TrackedTask(
            name=self._task_name(task),
            created_at=datetime.now(UTC),
            metadata=dict(metadata) if metadata is not None else None,
        )
        self._tasks[task] = record
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        record = self._tasks.pop(task, None)
        if record is None:
            return

        state = self._task_state(task)
        if state == "failed":
            try:
                exc = task.exception()
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Task %s failed but exception retrieval raised", record.name)
            else:
                self._logger.error(
                    "Background task %s failed: %s",
                    record.name or "<unnamed>",
                    exc,
                    exc_info=True,
                )
        elif state == "cancelled":
            self._logger.debug("Background task %s cancelled", record.name)
        else:
            self._logger.debug("Background task %s completed", record.name)

    def _task_state(self, task: asyncio.Task[Any]) -> str:
        if task.cancelled():
            return "cancelled"
        if not task.done():
            return "running"
        try:
            exc = task.exception()
        except Exception:  # pragma: no cover - defensive
            return "failed"
        return "failed" if exc else "finished"

    def _task_name(self, task: asyncio.Task[Any]) -> str | None:
        getter = getattr(task, "get_name", None)
        if callable(getter):
            try:
                return getter()
            except Exception:  # pragma: no cover - defensive
                return None
        return None
