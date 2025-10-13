"""Async task supervision utilities for runtime background workloads."""

from __future__ import annotations

import asyncio
import logging
from asyncio import AbstractEventLoop
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine, Mapping, MutableMapping, Sequence, TypeVar

__all__ = ["TaskSupervisor", "TaskSnapshot"]


@dataclass(slots=True, frozen=True)
class TaskSnapshot:
    """Snapshot describing a tracked task for diagnostics and summaries."""

    name: str | None
    state: str
    created_at: datetime
    age_seconds: float
    metadata: Mapping[str, Any] | None
    hang_timeout_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "age_seconds": self.age_seconds,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.hang_timeout_seconds is not None:
            payload["hang_timeout_seconds"] = self.hang_timeout_seconds
        return payload


@dataclass(slots=True)
class _TrackedTask:
    """Internal record describing an active task."""

    name: str | None
    created_at: datetime
    metadata: Mapping[str, Any] | None
    restart_limit: int | None
    restart_backoff: float
    restarts: int
    hang_timeout: float | None


_T = TypeVar("_T")


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
        self._installed_loop: AbstractEventLoop | None = None
        self._original_task_factory: Callable[..., asyncio.Task[Any]] | None = None
        self._loop_factory_metadata: Mapping[str, Any] | None = None
        self._task_counter: int = 0

    @property
    def active_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        """Return the currently tracked tasks that have not finished."""

        return tuple(self._tasks.keys())

    @property
    def active_count(self) -> int:
        """Return the number of active tracked tasks."""

        return len(self._tasks)

    @property
    def namespace(self) -> str:
        """Expose the namespace assigned to this supervisor."""

        return self._namespace

    def create(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        restart_callback: Callable[[], Coroutine[Any, Any, _T]] | None = None,
        max_restarts: int | None = 0,
        restart_backoff: float = 0.0,
        hang_timeout: float | None = None,
    ) -> asyncio.Task[_T]:
        """Create and track a background task."""

        if self._closing:
            raise RuntimeError("TaskSupervisor is shutting down; cannot create new tasks")

        if restart_callback is not None and not callable(restart_callback):
            raise TypeError("restart_callback must be callable when provided")
        if max_restarts is not None and max_restarts < 0:
            raise ValueError("max_restarts must be non-negative or None")

        restart_limit = max_restarts if max_restarts is not None else None
        effective_backoff = max(0.0, float(restart_backoff))
        effective_timeout: float | None
        if hang_timeout is None:
            effective_timeout = None
        else:
            try:
                timeout_value = float(hang_timeout)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise TypeError("hang_timeout must be convertible to float") from exc
            effective_timeout = timeout_value if timeout_value > 0 else None

        record_holder: list[_TrackedTask] = []

        async def _managed(initial_coro: Coroutine[Any, Any, _T]) -> _T:
            attempt = 0
            current = initial_coro
            task_name = name or "<unnamed>"
            while True:
                try:
                    if effective_timeout is not None:
                        return await asyncio.wait_for(current, timeout=effective_timeout)
                    return await current
                except asyncio.TimeoutError:
                    self._logger.error(
                        "Background task %s exceeded hang timeout %.2fs",
                        task_name,
                        effective_timeout,
                    )
                    if restart_callback is None or (
                        restart_limit is not None and attempt >= restart_limit
                    ):
                        raise
                    attempt += 1
                    if record_holder:
                        record_holder[0].restarts = attempt
                    limit_display = "∞" if restart_limit is None else str(restart_limit)
                    self._logger.error(
                        "Background task %s restarting after hang (attempt %s/%s)",
                        task_name,
                        attempt,
                        limit_display,
                    )
                    if effective_backoff:
                        try:
                            await asyncio.sleep(effective_backoff)
                        except asyncio.CancelledError:
                            raise
                    try:
                        current = restart_callback()
                    except Exception:  # pragma: no cover - restart factory failures
                        self._logger.exception(
                            "Failed to obtain restart coroutine for task %s", task_name
                        )
                        raise
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - restart logging path
                    if restart_callback is None or (
                        restart_limit is not None and attempt >= restart_limit
                    ):
                        raise
                    attempt += 1
                    if record_holder:
                        record_holder[0].restarts = attempt
                    limit_display = "∞" if restart_limit is None else str(restart_limit)
                    self._logger.error(
                        "Background task %s failed (attempt %s/%s); restarting",
                        task_name,
                        attempt,
                        limit_display,
                        exc_info=exc,
                    )
                    if effective_backoff:
                        try:
                            await asyncio.sleep(effective_backoff)
                        except asyncio.CancelledError:
                            raise
                    try:
                        current = restart_callback()
                    except Exception:  # pragma: no cover - restart factory failures
                        self._logger.exception(
                            "Failed to obtain restart coroutine for task %s", task_name
                        )
                        raise

        should_restart = restart_callback is not None and (
            restart_limit is None or restart_limit > 0
        )
        should_wrap = effective_timeout is not None or should_restart

        if should_wrap:
            managed_coro: Coroutine[Any, Any, _T] = _managed(coro)
        else:
            managed_coro = coro

        if should_restart:
            restart_meta_limit = restart_limit
            restart_meta_backoff = effective_backoff
        else:
            restart_meta_limit = None
            restart_meta_backoff = 0.0
            if not should_wrap:
                restart_callback = None

        if name is None:
            name = self._generate_task_name()

        task: asyncio.Task[_T] = asyncio.create_task(managed_coro, name=name)
        record = self._register(
            task,
            metadata,
            restart_limit=restart_meta_limit,
            restart_backoff=restart_meta_backoff,
            hang_timeout=effective_timeout,
        )
        if should_restart:
            record_holder.append(record)
        return task

    def install_loop_task_factory(
        self,
        *,
        loop: AbstractEventLoop | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Instrument ``loop`` so ``asyncio.create_task`` routes through the supervisor.

        When installed, any task spawned via ``loop.create_task`` (and therefore
        ``asyncio.create_task``) is automatically tracked by the supervisor.  A
        caller can provide ``metadata`` that will be attached to tasks that did
        not opt-in to supervision explicitly.  Installing the factory twice for
        the same loop is a no-op; attempting to install against a different
        loop without uninstalling first raises ``RuntimeError``.
        """

        if loop is None:
            loop = asyncio.get_running_loop()

        if self._installed_loop is loop:
            self._loop_factory_metadata = dict(metadata) if metadata is not None else None
            return
        if self._installed_loop is not None and self._installed_loop is not loop:
            raise RuntimeError(
                "TaskSupervisor loop task factory already installed on a different loop"
            )

        original_factory = loop.get_task_factory()

        def _supervised_factory(
            loop: AbstractEventLoop,
            coro: Coroutine[Any, Any, Any],
            *factory_args: object,
            **factory_kwargs: object,
        ) -> asyncio.Task[Any]:
            if original_factory is None:
                kwargs = dict(factory_kwargs)
                name_hint = kwargs.pop("name", None)
                context_obj = kwargs.pop("context", None)
                if kwargs:  # pragma: no cover - unexpected kwargs
                    self._logger.debug(
                        "Ignoring unsupported task factory kwargs: %s",
                        sorted(kwargs.keys()),
                    )
                task = asyncio.Task(coro, loop=loop, *factory_args)
                if context_obj is not None:
                    set_context = getattr(task, "set_context", None)
                    if callable(set_context):  # pragma: no cover - Py311+ specific hook
                        try:
                            set_context(context_obj)
                        except Exception:
                            self._logger.debug(
                                "Failed to apply task context for supervised task",
                                exc_info=True,
                            )
                if name_hint is not None:
                    try:
                        task.set_name(str(name_hint))
                    except Exception:
                        self._logger.debug(
                            "Failed to assign task name %s via loop factory", name_hint,
                            exc_info=True,
                        )
            else:
                task = original_factory(loop, coro, *factory_args, **factory_kwargs)

            try:
                if not self.is_tracked(task):
                    metadata_source = self._loop_factory_metadata
                    task_metadata = (
                        dict(metadata_source) if metadata_source is not None else None
                    )
                else:
                    task_metadata = None
                self.track(task, metadata=task_metadata)
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.debug(
                    "Failed to track task spawned via loop task factory", exc_info=True
                )
            return task

        loop.set_task_factory(_supervised_factory)
        self._installed_loop = loop
        self._original_task_factory = original_factory
        self._loop_factory_metadata = dict(metadata) if metadata is not None else None

    def uninstall_loop_task_factory(self) -> None:
        """Restore the loop's original task factory if previously installed."""

        loop = self._installed_loop
        if loop is None:
            return

        try:
            loop.set_task_factory(self._original_task_factory)
        except Exception:  # pragma: no cover - defensive best effort
            self._logger.debug(
                "Failed to restore original loop task factory", exc_info=True
            )
        finally:
            self._installed_loop = None
            self._original_task_factory = None
            self._loop_factory_metadata = None

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

        if task in self._tasks:
            record = self._tasks[task]
            if metadata and record.metadata is None:
                record.metadata = dict(metadata)
            elif metadata and isinstance(record.metadata, Mapping):
                merged = dict(record.metadata)
                merged.update(metadata)
                record.metadata = merged
            return task

        self._maybe_assign_default_name(task)
        self._register(task, metadata)
        return task

    def is_tracked(self, task: asyncio.Task[Any]) -> bool:
        """Return ``True`` when ``task`` is already tracked by the supervisor."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("TaskSupervisor.is_tracked expects an asyncio.Task instance")
        return task in self._tasks

    def describe(self) -> list[dict[str, Any]]:
        """Return lightweight metadata for active tasks."""

        now = datetime.now(UTC)
        snapshots: list[dict[str, Any]] = []
        for task, record in self._tasks.items():
            snapshot = TaskSnapshot(
                name=record.name,
                state=self._task_state(task),
                created_at=record.created_at,
                age_seconds=max(0.0, (now - record.created_at).total_seconds()),
                metadata=record.metadata,
                hang_timeout_seconds=record.hang_timeout,
            )
            payload = snapshot.as_dict()
            if record.restart_limit not in (None, 0):
                payload["restart_limit"] = record.restart_limit
            if record.restart_backoff:
                payload["restart_backoff_seconds"] = record.restart_backoff
            if record.restarts:
                payload["restarts"] = record.restarts
            snapshots.append(payload)
        return snapshots

    async def cancel_all(self) -> None:
        """Cancel tracked tasks and wait for them to exit."""

        if self._closing:
            return

        self.uninstall_loop_task_factory()

        tasks: Sequence[asyncio.Task[Any]] = tuple(self._tasks.keys())
        task_records = {task: self._tasks.get(task) for task in tasks}
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
            stalled: list[tuple[str | None, Mapping[str, Any] | None, str]] = []
            for task in tasks:
                record = task_records.get(task)
                name = record.name if record is not None else self._task_name(task)
                metadata = record.metadata if record is not None else None
                stalled.append((name, metadata, self._task_state(task)))

            for name, metadata, state in stalled:
                safe_metadata = dict(metadata) if metadata is not None else {}
                self._logger.error(
                    "Background task %s failed to stop within %.2fs (state=%s metadata=%s)",
                    name or "<unnamed>",
                    self._cancel_timeout,
                    state,
                    safe_metadata,
                )

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
        *,
        restart_limit: int | None = None,
        restart_backoff: float = 0.0,
        hang_timeout: float | None = None,
    ) -> _TrackedTask:
        record = _TrackedTask(
            name=self._task_name(task),
            created_at=datetime.now(UTC),
            metadata=dict(metadata) if metadata is not None else None,
            restart_limit=restart_limit,
            restart_backoff=restart_backoff,
            restarts=0,
            hang_timeout=hang_timeout,
        )
        self._tasks[task] = record
        task.add_done_callback(self._on_task_done)
        return record

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

    def _generate_task_name(self) -> str:
        self._task_counter += 1
        namespace = self._namespace or "task"
        return f"{namespace}-task-{self._task_counter:05d}"

    def _maybe_assign_default_name(self, task: asyncio.Task[Any]) -> None:
        current_name = self._task_name(task)
        if not self._should_replace_task_name(current_name):
            return

        new_name = self._generate_task_name()
        setter = getattr(task, "set_name", None)
        if callable(setter):
            try:
                setter(new_name)
            except Exception:  # pragma: no cover - defensive best effort
                self._logger.debug(
                    "Failed to assign generated task name", exc_info=True
                )

    @staticmethod
    def _should_replace_task_name(name: str | None) -> bool:
        if not name:
            return True
        if not isinstance(name, str):
            return True
        if name.startswith("Task-") and name[5:].isdigit():
            return True
        return False
