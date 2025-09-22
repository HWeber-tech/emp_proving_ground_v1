"""FIX integration pilot that supervises broker lifecycles and telemetry."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from src.runtime.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixPilotState:
    """Snapshot of the FIX pilot runtime state."""

    sessions_started: bool
    sensory_running: bool
    broker_running: bool
    queue_metrics: Mapping[str, Mapping[str, int]]
    active_orders: int
    last_order: Mapping[str, Any] | None
    compliance_summary: Mapping[str, Any] | None
    risk_summary: Mapping[str, Any] | None
    dropcopy_running: bool
    dropcopy_backlog: int
    last_dropcopy_event: Mapping[str, Any] | None
    dropcopy_reconciliation: Mapping[str, Any] | None
    timestamp: datetime


class FixIntegrationPilot:
    """Co-ordinates FIX connection manager, sensory organ, and broker lifecycle."""

    def __init__(
        self,
        *,
        connection_manager: Any,
        sensory_organ: Any,
        broker_interface: Any,
        task_supervisor: TaskSupervisor,
        event_bus: Any,
        compliance_monitor: Any | None = None,
        trading_manager: Any | None = None,
        dropcopy_listener: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.connection_manager = connection_manager
        self.sensory_organ = sensory_organ
        self.broker_interface = broker_interface
        self.task_supervisor = task_supervisor
        self.event_bus = event_bus
        self.compliance_monitor = compliance_monitor
        self.trading_manager = trading_manager
        self.dropcopy_listener = dropcopy_listener
        self._logger = logger or logging.getLogger(__name__)
        self._running = False
        self._sessions_started = False

    async def start(self) -> None:
        if self._running:
            return
        self._logger.info("🚦 Starting FIX integration pilot")
        self._sessions_started = bool(self.connection_manager.start_sessions())
        self._bind_message_queues()
        self._refresh_initiator()
        await self._start_component(self.sensory_organ)
        await self._start_component(self.broker_interface)
        await self._start_component(self.dropcopy_listener)
        self._running = True
        self._logger.info("🎯 FIX integration pilot running")

    async def stop(self) -> None:
        if not self._running and not self._sessions_started:
            self.connection_manager.stop_sessions()
            return
        self._logger.info("🛑 Stopping FIX integration pilot")
        await self._stop_component(self.dropcopy_listener)
        await self._stop_component(self.broker_interface)
        await self._stop_component(self.sensory_organ)
        self.connection_manager.stop_sessions()
        self._running = False
        self._logger.info("✅ FIX integration pilot stopped")

    def snapshot(self) -> FixPilotState:
        queue_metrics: dict[str, Mapping[str, int]] = {}
        for session in ("price", "trade"):
            adapter = self.connection_manager.get_application(session)
            metrics = None
            if adapter is not None and hasattr(adapter, "get_queue_metrics"):
                try:
                    metrics = adapter.get_queue_metrics()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Failed to read %s queue metrics", session, exc_info=True)
            if metrics:
                queue_metrics[session] = dict(metrics)

        active_orders = 0
        last_order: Mapping[str, Any] | None = None
        if hasattr(self.broker_interface, "get_all_orders"):
            try:
                orders = self.broker_interface.get_all_orders()
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Unable to fetch broker orders", exc_info=True)
            else:
                if isinstance(orders, Mapping):
                    active_orders = len(orders)
                    if orders:
                        _, last_order = next(reversed(list(orders.items())))
                elif isinstance(orders, list):
                    active_orders = len(orders)
                    if orders:
                        last_order = orders[-1]

        compliance_summary: Mapping[str, Any] | None = None
        if self.compliance_monitor is not None and hasattr(self.compliance_monitor, "summary"):
            try:
                compliance_summary = self.compliance_monitor.summary()
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Compliance summary failed", exc_info=True)

        risk_summary: Mapping[str, Any] | None = None
        if self.trading_manager is not None and hasattr(
            self.trading_manager, "get_execution_stats"
        ):
            try:
                stats = self.trading_manager.get_execution_stats()
                risk_summary = dict(stats)
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Risk stats fetch failed", exc_info=True)

        dropcopy_running = False
        dropcopy_backlog = 0
        last_dropcopy: Mapping[str, Any] | None = None
        dropcopy_reconciliation: Mapping[str, Any] | None = None
        listener = self.dropcopy_listener
        if listener is not None:
            dropcopy_running = bool(getattr(listener, "running", False))
            backlog_fn = getattr(listener, "get_backlog", None)
            if callable(backlog_fn):
                try:
                    dropcopy_backlog = int(backlog_fn())
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Drop-copy backlog read failed", exc_info=True)
            last_event_fn = getattr(listener, "get_last_event", None)
            if callable(last_event_fn):
                try:
                    last_event = last_event_fn()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Drop-copy last-event fetch failed", exc_info=True)
                else:
                    if isinstance(last_event, Mapping):
                        last_dropcopy = dict(last_event)
            reconciliation_fn = getattr(listener, "reconciliation_summary", None)
            if callable(reconciliation_fn):
                try:
                    reconciliation = reconciliation_fn()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Drop-copy reconciliation failed", exc_info=True)
                else:
                    if isinstance(reconciliation, Mapping):
                        dropcopy_reconciliation = dict(reconciliation)

        return FixPilotState(
            sessions_started=self._sessions_started,
            sensory_running=bool(getattr(self.sensory_organ, "running", False)),
            broker_running=bool(getattr(self.broker_interface, "running", False)),
            queue_metrics=queue_metrics,
            active_orders=active_orders,
            last_order=dict(last_order) if isinstance(last_order, Mapping) else None,
            compliance_summary=dict(compliance_summary)
            if isinstance(compliance_summary, Mapping)
            else None,
            risk_summary=dict(risk_summary) if isinstance(risk_summary, Mapping) else None,
            dropcopy_running=dropcopy_running,
            dropcopy_backlog=dropcopy_backlog,
            last_dropcopy_event=last_dropcopy,
            dropcopy_reconciliation=dropcopy_reconciliation,
            timestamp=datetime.now(tz=UTC),
        )

    # ------------------------------------------------------------------
    def _bind_message_queues(self) -> None:
        price_queue = getattr(self.sensory_organ, "price_queue", None)
        trade_queue = getattr(self.broker_interface, "trade_queue", None)
        if price_queue is not None:
            adapter = self.connection_manager.get_application("price")
            if adapter is not None and hasattr(adapter, "set_message_queue"):
                try:
                    adapter.set_message_queue(price_queue)
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Failed to bind price queue", exc_info=True)
        if trade_queue is not None:
            adapter = self.connection_manager.get_application("trade")
            if adapter is not None and hasattr(adapter, "set_message_queue"):
                try:
                    adapter.set_message_queue(trade_queue)
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Failed to bind trade queue", exc_info=True)
        listener = self.dropcopy_listener
        dropcopy_queue = getattr(listener, "dropcopy_queue", None)
        if dropcopy_queue is not None:
            adapter = self.connection_manager.get_application("dropcopy")
            if adapter is not None and hasattr(adapter, "set_message_queue"):
                try:
                    adapter.set_message_queue(dropcopy_queue)
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug("Failed to bind drop-copy queue", exc_info=True)

    def _refresh_initiator(self) -> None:
        initiator = self.connection_manager.get_initiator("trade")
        if initiator is None:
            return
        try:
            setattr(self.broker_interface, "fix_initiator", initiator)
        except Exception:  # pragma: no cover - diagnostics only
            self._logger.debug("Failed to refresh broker initiator", exc_info=True)

    async def _start_component(self, component: Any) -> None:
        if component is None:
            return
        start_method = getattr(component, "start", None)
        if start_method is None:
            return
        try:
            result = start_method()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover - start should surface errors
            self._logger.exception("Failed to start component %s", component.__class__.__name__)
            raise
        self._register_component_tasks(component)

    async def _stop_component(self, component: Any) -> None:
        if component is None:
            return
        stop_method = getattr(component, "stop", None)
        if stop_method is None:
            return
        try:
            result = stop_method()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover - diagnostics only
            self._logger.exception("Failed to stop component %s", component.__class__.__name__)

    def _register_component_tasks(self, component: Any) -> None:
        for attr in ("_price_task", "_trade_task"):
            task = getattr(component, attr, None)
            if isinstance(task, asyncio.Task):
                self.task_supervisor.track(task)
        drop_task = getattr(component, "_dropcopy_task", None)
        if isinstance(drop_task, asyncio.Task):
            self.task_supervisor.track(drop_task)


__all__ = [
    "FixIntegrationPilot",
    "FixPilotState",
]
