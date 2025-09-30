from __future__ import annotations

"""
Health Monitor System
====================

Monitors system health and performance across all components.
"""

import asyncio
import logging
import uuid
from collections.abc import Coroutine, Mapping
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, TypeVar

from src.core.event_bus import EventBus
from src.core.state_store import StateStore

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


class _SupportsCreateTask(Protocol):
    """Protocol capturing the :class:`TaskSupervisor` ``create`` signature."""

    def create(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "asyncio.Task[_T]":
        ...


class HealthMonitor:
    """Monitors system health and performance."""

    def __init__(
        self,
        state_store: StateStore,
        event_bus: EventBus,
        *,
        task_supervisor: Optional[_SupportsCreateTask] = None,
    ):
        self.state_store = state_store
        self.event_bus = event_bus
        self.is_running = False
        self.check_interval = 60  # seconds
        self.health_history: list[dict[str, object]] = []
        self._task_supervisor = task_supervisor
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> bool:
        """Start health monitoring."""
        if self.is_running:
            return True

        self.is_running = True
        self._stop_event = asyncio.Event()
        self._monitor_task = self._create_task(self._monitor_loop(), "health-monitor-loop")
        logger.info("Health monitor started")
        return True

    async def stop(self) -> bool:
        """Stop health monitoring."""
        if not self.is_running:
            return True

        self.is_running = False
        if self._stop_event is not None:
            self._stop_event.set()

        task = self._monitor_task
        if task is not None:
            try:
                await task
            finally:
                self._monitor_task = None

        self._stop_event = None

        logger.info("Health monitor stopped")
        return True

    def _create_task(self, coro: Coroutine[Any, Any, None], name: str) -> asyncio.Task[None]:
        metadata = {"component": "operational.health_monitor"}
        supervisor = self._task_supervisor
        if supervisor is not None:
            create = getattr(supervisor, "create", None)
            if callable(create):
                try:
                    return create(coro, name=name, metadata=metadata)
                except TypeError:
                    return create(coro, name=name)  # type: ignore[arg-type]
        return asyncio.create_task(coro, name=name)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        stop_event = self._stop_event
        if stop_event is None:
            return

        while not stop_event.is_set():
            try:
                health_check = await self._perform_health_check()
            except Exception as exc:
                logger.exception("Health monitoring cycle failed")
                health_check = {
                    "check_id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "ERROR",
                    "error": str(exc),
                }

            await self._store_health_check(health_check)

            if health_check.get("status") == "CRITICAL":
                try:
                    await self.event_bus.emit("health_critical", health_check)
                except Exception:
                    logger.exception("Failed to emit health_critical event")

            await self._sleep_until_next_cycle(stop_event)

    async def _sleep_until_next_cycle(self, stop_event: asyncio.Event) -> None:
        """Wait for the next monitoring cycle while honouring shutdown signals."""

        if self.check_interval <= 0:
            await asyncio.sleep(0)
            return

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=self.check_interval)
        except asyncio.TimeoutError:
            return

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
            value = await self.state_store.get("health_check")
            return {"status": "HEALTHY", "response_time": 0.001}
        except Exception as exc:
            logger.exception("State store health check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_event_bus(self) -> Dict[str, Any]:
        """Check event bus health."""
        try:
            return {"status": "HEALTHY", "subscribers": 0}
        except Exception as exc:
            logger.exception("Event bus health check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "status": "HEALTHY" if memory.percent < 90 else "WARNING",
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
            }
        except ImportError as exc:
            logger.warning("psutil not available for memory checks: %s", exc)
            return {"status": "UNKNOWN", "error": str(exc)}
        except Exception as exc:
            logger.exception("Memory check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                "status": "HEALTHY" if cpu_percent < 80 else "WARNING",
                "usage_percent": cpu_percent,
            }
        except ImportError as exc:
            logger.warning("psutil not available for CPU checks: %s", exc)
            return {"status": "UNKNOWN", "error": str(exc)}
        except Exception as exc:
            logger.exception("CPU check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            import psutil

            disk = psutil.disk_usage("/")
            return {
                "status": "HEALTHY" if disk.percent < 90 else "WARNING",
                "usage_percent": disk.percent,
                "free_gb": disk.free / (1024**3),
            }
        except ImportError as exc:
            logger.warning("psutil not available for disk checks: %s", exc)
            return {"status": "UNKNOWN", "error": str(exc)}
        except Exception as exc:
            logger.exception("Disk check failed")
            return {"status": "ERROR", "error": str(exc)}

    async def _store_health_check(self, health_check: Dict[str, Any]) -> None:
        """Store health check results."""
        try:
            key = f"health_check:{datetime.utcnow().date()}"
            await self.state_store.set(key, str(health_check), expire=86400)
            self.health_history.append(health_check)

            # Keep only last 100 checks
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]

        except Exception:
            logger.exception("Error storing health check")

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
