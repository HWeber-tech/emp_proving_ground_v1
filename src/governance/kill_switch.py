"""Asynchronous monitoring utilities for the global kill-switch."""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

__all__ = ["KillSwitchMonitor"]


KillSwitchCallback = Callable[[Path], Awaitable[None] | None]


class KillSwitchMonitor:
    """Poll a kill-switch file and invoke a callback when it is engaged.

    The trading runtime exposes a *kill-switch* mechanism implemented as a
    sentinel file.  When the file exists, the runtime must halt immediately and
    proceed with a graceful shutdown.  ``KillSwitchMonitor`` provides a small
    asynchronous helper that watches the configured path and invokes a callback
    exactly once when the kill-switch is detected.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        poll_interval: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        try:
            interval = float(poll_interval)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("poll_interval must be a number") from exc
        if interval <= 0:
            raise ValueError("poll_interval must be positive")

        self._path = Path(path).expanduser()
        self._poll_interval = interval
        self._logger = logger or logging.getLogger(__name__)
        self._triggered = False

    @property
    def path(self) -> Path:
        """Expose the monitored kill-switch path."""

        return self._path

    @property
    def triggered(self) -> bool:
        """Return ``True`` when the kill-switch has been engaged."""

        return self._triggered

    async def run(
        self,
        callback: KillSwitchCallback,
        *,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Monitor the kill-switch until triggered or ``stop_event`` is set.

        Parameters
        ----------
        callback:
            Awaitable or synchronous callable invoked when the kill-switch
            exists.  The ``Path`` for the kill-switch is provided as the sole
            argument.  The callback is awaited when it returns an awaitable.
        stop_event:
            Optional :class:`asyncio.Event` used to terminate monitoring when
            the runtime is shutting down for another reason.
        """

        if not callable(callback):
            raise TypeError("KillSwitchMonitor.run requires a callable callback")

        interval = self._poll_interval
        event = stop_event

        while True:
            if event is not None and event.is_set():
                return

            try:
                engaged = self._path.exists()
            except OSError as exc:  # pragma: no cover - OS specific, defensive
                self._logger.warning(
                    "Kill-switch probe failed for %s: %s", self._path, exc
                )
                engaged = False

            if engaged:
                if self._triggered:
                    return
                self._triggered = True
                self._logger.critical(
                    "Kill-switch engaged at %s; initiating emergency shutdown", self._path
                )
                try:
                    result = callback(self._path)
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # pragma: no cover - callback robustness
                    self._logger.exception("Kill-switch callback failed")
                return

            if event is None:
                await asyncio.sleep(interval)
            else:
                try:
                    await asyncio.wait_for(event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    continue
                if event.is_set():
                    return
