"""Health monitoring utilities with supervised async lifecycle support."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Optional, cast, TYPE_CHECKING

from src.core.event_bus import EventBus
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from src.runtime.task_supervisor import TaskSupervisor


class _PsutilUnavailableError(RuntimeError):
    """Raised when psutil cannot be imported for resource probes."""


def _import_psutil() -> ModuleType:
    """Import ``psutil`` lazily to keep optional dependency semantics."""

    try:
        return cast(ModuleType, import_module("psutil"))
    except ImportError as exc:
        raise _PsutilUnavailableError("psutil is not available") from exc


class HealthMonitor:
    """Monitors system health and performance."""

    def __init__(self, state_store: StateStore, event_bus: EventBus):
        self.state_store = state_store
        self.event_bus = event_bus
        self.is_running = False
        self.check_interval = 60  # seconds
        self.health_history: list[dict[str, object]] = []
        self._monitor_task: asyncio.Task[None] | None = None
        self._task_supervisor: TaskSupervisor | None = None

    async def start(self, *, task_supervisor: "TaskSupervisor | None" = None) -> bool:
        """Start health monitoring and optionally register with a supervisor."""
        if self.is_running:
            return True

        self.is_running = True
        if task_supervisor is not None:
            self._task_supervisor = task_supervisor

        metadata = {
            "component": "health-monitor",
            "interval_seconds": self.check_interval,
        }
        loop_coro = self._monitor_loop()
        if self._task_supervisor is not None:
            task = self._task_supervisor.create(
                loop_coro,
                name="health-monitor-loop",
                metadata=metadata,
            )
        else:
            task = asyncio.create_task(loop_coro, name="health-monitor-loop")

        self._monitor_task = task
        task.add_done_callback(self._on_monitor_task_done)
        logger.info("Health monitor started")
        return True

    async def stop(self) -> bool:
        """Stop health monitoring."""
        self.is_running = False
        task = self._monitor_task
        self._monitor_task = None
        if task is not None:
            try:
                if not task.done():
                    task.cancel()
                await task
            except asyncio.CancelledError:
                logger.debug("Health monitor loop cancelled during shutdown")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "Health monitor task failed during shutdown", exc_info=exc
                )
        logger.info("Health monitor stopped")
        return True

    def _on_monitor_task_done(self, task: asyncio.Task[None]) -> None:
        if self._monitor_task is task:
            self._monitor_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("Health monitor loop cancelled")
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Health monitor task failed")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                health_check = await self._perform_health_check()
                await self._store_health_check(health_check)

                if health_check.get("status") == "CRITICAL":
                    await self.event_bus.emit("health_critical", health_check)

                await asyncio.sleep(self.check_interval)

            except Exception as exc:
                logger.exception("Error in health monitoring loop", exc_info=exc)
                await asyncio.sleep(self.check_interval)

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        return {
            "check_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "HEALTHY",
            "components": {
                "state_store": await self._check_state_store(),
                "event_bus": await self._check_event_bus(),
                "memory": await self._check_memory(),
                "cpu": await self._check_cpu(),
                "disk": await self._check_disk(),
            },
        }

    async def _check_state_store(self) -> Dict[str, Any]:
        """Check state store health."""
        try:
            await self.state_store.set("health_check", "test")
            await self.state_store.get("health_check")
            return {"status": "HEALTHY", "response_time": 0.001}
        except Exception as exc:
            logger.warning("State store health check failed", exc_info=exc)
            return {"status": "ERROR", "error": str(exc)}

    async def _check_event_bus(self) -> Dict[str, Any]:
        """Check event bus health."""
        try:
            stats = self.event_bus.get_statistics()
        except AttributeError as exc:
            logger.error("Event bus missing statistics interface", exc_info=exc)
            return {"status": "ERROR", "error": str(exc)}
        except Exception as exc:
            logger.error("Event bus statistics retrieval failed", exc_info=exc)
            return {"status": "ERROR", "error": str(exc)}

        return {
            "status": "HEALTHY" if stats.running and stats.loop_running else "WARNING",
            "subscribers": stats.subscriber_count,
            "queue_size": stats.queue_size,
            "dropped_events": stats.dropped_events,
        }

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            psutil = _import_psutil()
            memory = psutil.virtual_memory()
        except _PsutilUnavailableError as exc:
            logger.warning("psutil unavailable for memory probe", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}
        except Exception as exc:
            logger.error("Failed to sample memory usage", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}

        return {
            "status": "HEALTHY" if memory.percent < 90 else "WARNING",
            "usage_percent": memory.percent,
            "available_gb": memory.available / (1024**3),
        }

    async def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            psutil = _import_psutil()
            cpu_percent = psutil.cpu_percent(interval=1)
        except _PsutilUnavailableError as exc:
            logger.warning("psutil unavailable for CPU probe", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}
        except Exception as exc:
            logger.error("Failed to sample CPU usage", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}

        return {
            "status": "HEALTHY" if cpu_percent < 80 else "WARNING",
            "usage_percent": cpu_percent,
        }

    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            psutil = _import_psutil()
            disk = psutil.disk_usage("/")
        except _PsutilUnavailableError as exc:
            logger.warning("psutil unavailable for disk probe", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}
        except Exception as exc:
            logger.error("Failed to sample disk usage", exc_info=exc)
            return {"status": "UNKNOWN", "usage_percent": 0, "error": str(exc)}

        return {
            "status": "HEALTHY" if disk.percent < 90 else "WARNING",
            "usage_percent": disk.percent,
            "free_gb": disk.free / (1024**3),
        }

    async def _store_health_check(self, health_check: Dict[str, Any]) -> None:
        """Store health check results."""
        try:
            key = f"health_check:{datetime.utcnow().date()}"
            await self.state_store.set(key, str(health_check), expire=86400)
            self.health_history.append(health_check)

            # Keep only last 100 checks
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]

        except Exception as exc:
            logger.exception("Error storing health check", exc_info=exc)

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self.health_history:
            return {"status": "NO_DATA", "message": "No health checks performed"}

        latest = self.health_history[-1]
        healthy_checks = sum(1 for h in self.health_history[-10:] if h.get("status") == "HEALTHY")

        return {
            "status": latest.get("status"),
            "uptime_percent": (healthy_checks / min(10, len(self.health_history))) * 100,
            "last_check": latest.get("timestamp"),
            "total_checks": len(self.health_history),
        }


# Global instance
_health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor(state_store: StateStore, event_bus: EventBus) -> HealthMonitor:
    """Get or create global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(state_store, event_bus)
    return _health_monitor
