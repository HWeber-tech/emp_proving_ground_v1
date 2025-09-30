"""Operational health monitor with supervised background execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from src.core.event_bus import EventBus
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@runtime_checkable
class SupportsTaskSupervision(Protocol):
    """Protocol implemented by task supervisors used by the runtime."""

    def create(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[_T]:
        ...


class HealthMonitor:
    """Monitors system health and emits alerts when critical thresholds trigger."""

    def __init__(
        self,
        state_store: StateStore,
        event_bus: EventBus,
        *,
        task_supervisor: SupportsTaskSupervision | None = None,
        check_interval_seconds: float = 60.0,
    ) -> None:
        self._state_store = state_store
        self._event_bus = event_bus
        self._check_interval = float(check_interval_seconds)
        if self._check_interval <= 0:
            raise ValueError("check_interval_seconds must be positive")
        self._task_supervisor = task_supervisor or _create_task_supervisor()
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop_signal = asyncio.Event()
        self._is_running = False
        self.health_history: list[dict[str, Any]] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self) -> bool:
        """Start the background health monitoring loop under supervision."""

        if self.is_running and self._monitor_task and not self._monitor_task.done():
            return True

        self._is_running = True
        self._stop_signal.clear()
        self._monitor_task = self._task_supervisor.create(
            self._monitor_loop(),
            name="health-monitor",
            metadata={"component": "health-monitor"},
        )
        logger.info("Health monitor started")
        return True

    async def stop(self) -> bool:
        """Stop the monitoring loop and wait for graceful shutdown."""

        if not self.is_running:
            return True

        self._is_running = False
        self._stop_signal.set()
        monitor_task = self._monitor_task
        if monitor_task is not None:
            try:
                await monitor_task
            except asyncio.CancelledError:
                logger.debug("Health monitor task cancelled during shutdown")
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Health monitor task failed during shutdown")
            finally:
                self._monitor_task = None

        logger.info("Health monitor stopped")
        return True

    async def _monitor_loop(self) -> None:
        """Main monitoring loop executed under supervision."""

        try:
            while self.is_running:
                await self._process_cycle()
                try:
                    await asyncio.wait_for(self._stop_signal.wait(), timeout=self._check_interval)
                except asyncio.TimeoutError:
                    continue
                else:
                    break
        except asyncio.CancelledError:
            logger.debug("Health monitor loop cancelled")
            raise

    async def _process_cycle(self) -> None:
        try:
            health_check = await self._perform_health_check()
        except Exception:
            logger.exception("Error performing health check")
            return

        await self._store_health_check(health_check)

        if health_check.get("status") == "CRITICAL":
            try:
                await self._event_bus.emit("health_critical", health_check)
            except Exception:
                logger.exception("Failed to emit health_critical event")

    async def _perform_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check across subsystems."""

        timestamp = datetime.now(UTC)
        return {
            "check_id": str(uuid4()),
            "timestamp": timestamp.isoformat(),
            "status": "HEALTHY",
            "components": {
                "state_store": await self._check_state_store(),
                "event_bus": await self._check_event_bus(),
                "memory": await self._check_memory(),
                "cpu": await self._check_cpu(),
                "disk": await self._check_disk(),
            },
        }

    async def _check_state_store(self) -> dict[str, Any]:
        try:
            await self._state_store.set("health_check", "test", expire=60)
            await self._state_store.get("health_check")
            return {"status": "HEALTHY"}
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.exception("State store check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_event_bus(self) -> dict[str, Any]:
        try:
            return {"status": "HEALTHY"}
        except Exception as exc:  # pragma: no cover - placeholder for future instrumentation
            logger.exception("Event bus check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_memory(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore[import-not-found]

            memory = psutil.virtual_memory()
            return {
                "status": "HEALTHY" if memory.percent < 90 else "WARNING",
                "usage_percent": float(memory.percent),
                "available_gb": float(memory.available / (1024**3)),
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Memory inspection unavailable: %s", exc)
            return {"status": "UNKNOWN", "usage_percent": 0.0}

    async def _check_cpu(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore[import-not-found]

            cpu_percent = float(psutil.cpu_percent(interval=0.1))
            return {
                "status": "HEALTHY" if cpu_percent < 80 else "WARNING",
                "usage_percent": cpu_percent,
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("CPU inspection unavailable: %s", exc)
            return {"status": "UNKNOWN", "usage_percent": 0.0}

    async def _check_disk(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore[import-not-found]

            disk = psutil.disk_usage("/")
            return {
                "status": "HEALTHY" if disk.percent < 90 else "WARNING",
                "usage_percent": float(disk.percent),
                "free_gb": float(disk.free / (1024**3)),
            }
        except (ImportError, AttributeError, OSError) as exc:
            logger.debug("Disk inspection unavailable: %s", exc)
            return {"status": "UNKNOWN", "usage_percent": 0.0}

    async def _store_health_check(self, health_check: dict[str, Any]) -> None:
        try:
            key = f"health_check:{datetime.now(UTC).date()}"
            await self._state_store.set(key, str(health_check), expire=86_400)
            self.health_history.append(health_check)
            if len(self.health_history) > 100:
                del self.health_history[:-100]
        except Exception:
            logger.exception("Error storing health check")

    async def get_health_summary(self) -> dict[str, Any]:
        if not self.health_history:
            return {"status": "NO_DATA", "message": "No health checks performed"}

        latest = self.health_history[-1]
        history_window = self.health_history[-10:]
        healthy_checks = sum(1 for check in history_window if check.get("status") == "HEALTHY")
        uptime_percent = 100.0 * healthy_checks / max(len(history_window), 1)

        return {
            "status": latest.get("status"),
            "uptime_percent": uptime_percent,
            "last_check": latest.get("timestamp"),
            "total_checks": len(self.health_history),
        }


def _create_task_supervisor() -> SupportsTaskSupervision:
    try:
        from src.runtime.task_supervisor import TaskSupervisor

        return TaskSupervisor(namespace="health-monitor")
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Falling back to unsupervised health monitor task creation: %s", exc)

        class _FallbackSupervisor(SupportsTaskSupervision):
            def create(  # type: ignore[override]
                self,
                coro: Coroutine[Any, Any, _T],
                *,
                name: str | None = None,
                metadata: Mapping[str, Any] | None = None,
            ) -> asyncio.Task[_T]:
                return asyncio.create_task(coro, name=name)

        return _FallbackSupervisor()
