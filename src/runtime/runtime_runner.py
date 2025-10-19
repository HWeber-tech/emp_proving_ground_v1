"""Helpers for running runtime applications with supervised lifecycles."""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from contextlib import suppress
from typing import Iterable

from src.runtime.runtime_builder import RuntimeApplication
from src.runtime.task_supervisor import TaskSupervisor

__all__ = ["run_runtime_application"]


async def run_runtime_application(
    runtime_app: RuntimeApplication,
    *,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
    namespace: str = "runtime.runner",
    register_signals: bool = True,
    supervisor: TaskSupervisor | None = None,
) -> None:
    """Run ``runtime_app`` under a :class:`TaskSupervisor` with signal handling.

    Parameters
    ----------
    runtime_app:
        Application container returned by :func:`build_professional_runtime_application`.
    timeout:
        Optional timeout (seconds) after which the runtime is cancelled.
    logger:
        Logger used for informational messages; defaults to a module logger.
    namespace:
        Namespace assigned to the task supervisor, aiding diagnostics.
    register_signals:
        Whether to register ``SIGINT``/``SIGTERM`` handlers that trigger a
        graceful shutdown.  Disabled automatically when the platform does not
        support custom signal handlers.
    supervisor:
        Optional pre-created :class:`TaskSupervisor` instance.  When omitted a
        fresh supervisor is created and disposed automatically.
    """

    loop = asyncio.get_running_loop()
    managed_logger = logger or logging.getLogger(f"{__name__}.{namespace}")
    app_supervisor = getattr(runtime_app, "task_supervisor", None)
    if supervisor is not None:
        managed_supervisor = supervisor
        owns_supervisor = False
    elif isinstance(app_supervisor, TaskSupervisor):
        managed_supervisor = app_supervisor
        owns_supervisor = False
    else:
        managed_supervisor = TaskSupervisor(namespace=namespace, logger=managed_logger)
        owns_supervisor = True

    bind_supervisor = getattr(runtime_app, "bind_task_supervisor", None)
    if callable(bind_supervisor):
        bind_supervisor(managed_supervisor)

    stop_event = asyncio.Event()

    def _trigger_stop() -> None:
        stop_event.set()

    registered_signals: list[signal.Signals] = []
    if register_signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _trigger_stop)
            except (NotImplementedError, ValueError):  # pragma: no cover - Windows / nested loop
                continue
            else:
                registered_signals.append(sig)

    run_task = managed_supervisor.create(runtime_app.run(), name="runtime-app-run")
    waiters: set[asyncio.Task[object]] = {run_task}

    stop_task = managed_supervisor.create(stop_event.wait(), name="runtime-stop-event")
    waiters.add(stop_task)

    timeout_task: asyncio.Task[object] | None = None
    if timeout is not None:
        timeout_task = managed_supervisor.create(
            asyncio.sleep(timeout),
            name="runtime-timeout",
        )
        waiters.add(timeout_task)

    try:
        done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

        if run_task in done:
            await _cancel_pending(pending)
            return

        if stop_task in done and stop_event.is_set():
            managed_logger.info("Shutdown signal received; cancelling runtime workloads")
        elif timeout_task is not None and timeout_task in done:
            managed_logger.info("Runtime timeout reached after %ss; cancelling workloads", timeout)

        run_task.cancel()
        await _cancel_pending(pending)
        with suppress(asyncio.CancelledError):
            await run_task
    finally:
        if owns_supervisor or managed_supervisor.active_count:
            await managed_supervisor.cancel_all()
        for sig in registered_signals:
            with suppress(NotImplementedError):  # pragma: no cover - mirrors add_signal_handler
                loop.remove_signal_handler(sig)
        try:
            await runtime_app.shutdown()
        except asyncio.CancelledError:
            with suppress(Exception):
                await runtime_app.shutdown()
        except Exception:
            pass

        if not getattr(runtime_app, "_shutdown_callbacks_executed", False):
            for callback in reversed(runtime_app.shutdown_callbacks):
                try:
                    result = callback()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # pragma: no cover - defensive guard
                    managed_logger.exception(
                        "Shutdown callback %r failed during runner fallback", callback
                    )


async def _cancel_pending(tasks: Iterable[asyncio.Task[object]]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
