"""Composable runtime assembly for the EMP Professional Predator."""
from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence, TYPE_CHECKING

from src.core.event_bus import EventBus
from src.governance.safety_manager import SafetyManager
from src.governance.system_config import ConnectionProtocol, SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor
from src.trading.integration.fix_broker_interface import FIXBrokerInterface

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from src.sensory.signals import SensorSignal


CleanupCallback = Callable[[], Awaitable[None] | None]


class MarketDataSensor(Protocol):
    """Protocol describing the narrow interface used by Tier-0 ingestion."""

    def process(self, df: "pd.DataFrame") -> Sequence["SensorSignal"]:
        ...


class ProfessionalPredatorApp:
    """High-level application wrapper with explicit lifecycle management."""

    def __init__(
        self,
        *,
        config: SystemConfig,
        event_bus: EventBus,
        sensory_organ: Optional[FIXSensoryOrgan],
        broker_interface: Optional[FIXBrokerInterface],
        fix_connection_manager: Optional[FIXConnectionManager],
        sensors: Mapping[str, MarketDataSensor],
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.sensory_organ = sensory_organ
        self.broker_interface = broker_interface
        self.fix_connection_manager = fix_connection_manager
        self._sensors = dict(sensors)

        self._logger = logging.getLogger(__name__)
        self._cleanup_callbacks: list[CleanupCallback] = []
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._stop_event = asyncio.Event()
        self._started = False
        self._closed = False
        self._start_time: datetime | None = None
        self._shutdown_time: datetime | None = None

    async def __aenter__(self) -> "ProfessionalPredatorApp":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    @property
    def sensors(self) -> Mapping[str, MarketDataSensor]:
        """Read-only view of configured sensors."""

        return MappingProxyType(self._sensors)

    def add_cleanup_callback(self, callback: CleanupCallback) -> None:
        self._cleanup_callbacks.append(callback)

    def create_background_task(self, coro: Awaitable[Any], *, name: Optional[str] = None) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def register_background_task(self, task: asyncio.Task[Any]) -> None:
        """Track an externally-created background task for managed shutdown."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("register_background_task expects an asyncio.Task")
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _register_component_tasks(self, component: Any) -> None:
        """Capture known background tasks emitted by runtime components."""

        for attr in ("_price_task", "_trade_task"):
            task = getattr(component, attr, None)
            if isinstance(task, asyncio.Task):
                self.register_background_task(task)

    async def _start_component(self, component: Any) -> None:
        if component is None:
            return

        start_method = getattr(component, "start", None)
        if start_method is None:
            return

        try:
            result = start_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception("Error starting component %s", component.__class__.__name__)
            raise
        else:
            self._register_component_tasks(component)

    async def _stop_component(self, component: Any) -> None:
        if component is None:
            return

        stop_method = getattr(component, "stop", None)
        if stop_method is None:
            return

        try:
            result = stop_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception("Error stopping component %s", component.__class__.__name__)

    async def _activate_components(self) -> None:
        await self._start_component(self.sensory_organ)
        await self._start_component(self.broker_interface)

    async def _deactivate_components(self) -> None:
        await self._stop_component(self.broker_interface)
        await self._stop_component(self.sensory_organ)

    async def start(self) -> None:
        if self._started:
            return

        self._logger.info("🚀 Initializing EMP v4.0 Professional Predator")
        self._logger.info("✅ Configuration loaded: %s", self.config.to_dict())
        self._logger.info("🔧 Protocol: %s", self.config.connection_protocol.value)
        self._logger.info("🧰 Run mode: %s", self.config.run_mode.value)
        self._logger.info("🏷️ Tier selected: %s", self.config.tier.value)
        if self.sensory_organ:
            self._logger.info("✅ %s ready", self.sensory_organ.__class__.__name__)
        if self.broker_interface:
            self._logger.info("✅ %s ready", self.broker_interface.__class__.__name__)

        self._stop_event = asyncio.Event()
        self._started = True
        self._closed = False
        self._start_time = datetime.now()
        self._shutdown_time = None

        await self._activate_components()
        self._logger.info("🎉 Professional Predator initialization complete")

    async def run_forever(self, heartbeat_seconds: float = 60.0) -> None:
        """Block until shutdown, emitting debug heartbeats."""

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                self._logger.debug("Heartbeat check - system alive")

    def request_shutdown(self) -> None:
        self._stop_event.set()

    async def shutdown(self) -> None:
        if self._closed:
            return

        self.request_shutdown()

        await self._deactivate_components()

        if self._background_tasks:
            tasks = tuple(self._background_tasks)
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for task, result in zip(tasks, results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    self._logger.warning(
                        "Background task %r exited with %s: %s",
                        task.get_name() if hasattr(task, "get_name") else task,
                        result.__class__.__name__,
                        result,
                    )

        for callback in reversed(self._cleanup_callbacks):
            try:
                outcome = callback()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Error during shutdown callback %s", callback)

        self._background_tasks.clear()
        self._shutdown_time = datetime.now()
        self._closed = True
        self._started = False
        self._logger.info("✅ Professional Predator shutdown complete")

    def summary(self) -> Dict[str, Any]:
        status: str
        if self._closed:
            status = "STOPPED"
        elif self._started:
            status = "RUNNING"
        else:
            status = "INITIALIZED"

        uptime_seconds: float = 0.0
        if self._start_time:
            end_time = self._shutdown_time or datetime.now()
            uptime_seconds = max((end_time - self._start_time).total_seconds(), 0.0)

        components: Dict[str, Any] = {
            "sensory_organ": self.sensory_organ.__class__.__name__ if self.sensory_organ else None,
            "broker_interface": self.broker_interface.__class__.__name__ if self.broker_interface else None,
            "fix_manager": "FIXConnectionManager" if self.fix_connection_manager else None,
            "sensors": sorted(self._sensors.keys()),
        }

        if self.sensory_organ and hasattr(self.sensory_organ, "running"):
            components["sensory_running"] = bool(getattr(self.sensory_organ, "running"))
        if self.broker_interface and hasattr(self.broker_interface, "running"):
            components["broker_running"] = bool(getattr(self.broker_interface, "running"))

        queue_metrics: Dict[str, Dict[str, int]] = {}
        if self.fix_connection_manager:
            price_app = self.fix_connection_manager.get_application("price")
            trade_app = self.fix_connection_manager.get_application("trade")
            if price_app and hasattr(price_app, "get_queue_metrics"):
                queue_metrics["price"] = dict(price_app.get_queue_metrics())
            if trade_app and hasattr(trade_app, "get_queue_metrics"):
                queue_metrics["trade"] = dict(trade_app.get_queue_metrics())
        if queue_metrics:
            components["queue_metrics"] = queue_metrics

        return {
            "version": "4.0",
            "protocol": self.config.connection_protocol.value,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "background_tasks": len(self._background_tasks),
            "cleanup_callbacks": len(self._cleanup_callbacks),
            "components": components,
        }


def _ensure_fix_components(
    config: SystemConfig,
    event_bus: EventBus,
) -> tuple[FIXConnectionManager, FIXSensoryOrgan, FIXBrokerInterface]:
    logger = logging.getLogger(__name__)
    logger.info("🔧 Setting up LIVE components using '%s' protocol", config.connection_protocol.value)
    logger.info("🎯 Configuring FIX protocol components")

    fix_connection_manager = FIXConnectionManager(config)
    fix_connection_manager.start_sessions()

    price_queue: asyncio.Queue[Any] = asyncio.Queue()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()

    price_app = fix_connection_manager.get_application("price")
    if price_app:
        price_app.set_message_queue(price_queue)
    trade_app = fix_connection_manager.get_application("trade")
    if trade_app:
        trade_app.set_message_queue(trade_queue)

    sensory_organ = FIXSensoryOrgan(event_bus, price_queue, config)
    broker_interface = FIXBrokerInterface(
        event_bus,
        trade_queue,
        fix_connection_manager.get_initiator("trade"),
    )

    logger.info("✅ FIX components configured successfully")
    return fix_connection_manager, sensory_organ, broker_interface


async def build_professional_predator_app(
    *,
    config: Optional[SystemConfig] = None,
    event_bus: Optional[EventBus] = None,
) -> ProfessionalPredatorApp:
    """Assemble a ProfessionalPredatorApp with all mandatory dependencies."""

    cfg = config or SystemConfig.from_env()
    bus = event_bus or EventBus()

    # Enforce guardrails before instantiating live components
    SafetyManager.from_config(cfg).enforce()

    if cfg.connection_protocol != ConnectionProtocol.fix:
        raise ValueError(
            f"Unsupported connection protocol: {cfg.connection_protocol.value}. "
            "Set CONNECTION_PROTOCOL=fix and follow docs/fix_api guides."
        )

    fix_manager, sensory_organ, broker_interface = _ensure_fix_components(cfg, bus)

    sensors: Dict[str, MarketDataSensor] = {
        "why": WhySensor(),
        "what": WhatSensor(),
        "when": WhenSensor(),
    }

    app = ProfessionalPredatorApp(
        config=cfg,
        event_bus=bus,
        sensory_organ=sensory_organ,
        broker_interface=broker_interface,
        fix_connection_manager=fix_manager,
        sensors=sensors,
    )

    app.add_cleanup_callback(fix_manager.stop_sessions)
    return app
