import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.operations.fix_pilot import FixPilotComponent, FixPilotSnapshot, FixPilotStatus
from src.config.risk.risk_config import RiskConfig
from src.runtime.predator_app import ProfessionalPredatorApp
from src.trading.risk.risk_api import RISK_API_RUNBOOK, RiskApiError
from src.runtime.task_supervisor import TaskSupervisor
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.trading.integration.fix_broker_interface import FIXBrokerInterface


class _StubSensor:
    def process(self, df):  # pragma: no cover - simple stub
        return []


class _StubAppAdapter:
    def __init__(self, delivered: int) -> None:
        self._metrics = {"delivered": delivered, "dropped": 0}

    def set_message_queue(self, queue):  # pragma: no cover - not used in tests
        self.queue = queue

    def get_queue_metrics(self):
        return dict(self._metrics)


class _StubFixManager:
    def __init__(self) -> None:
        self.price_adapter = _StubAppAdapter(delivered=5)
        self.trade_adapter = _StubAppAdapter(delivered=7)
        self.stopped = 0

    def stop_sessions(self) -> None:
        self.stopped += 1

    def get_application(self, session: str):
        if session == "price":
            return self.price_adapter
        if session == "trade":
            return self.trade_adapter
        return None


class _StubSensory:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.running = False
        self._price_task = None

    async def start(self) -> None:
        self.started += 1
        self.running = True
        self._price_task = asyncio.create_task(asyncio.sleep(0.01))

    async def stop(self) -> None:
        self.stopped += 1
        self.running = False
        task = self._price_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._price_task = None


class _StubBroker:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.running = False
        self._trade_task = None

    async def start(self) -> None:
        self.started += 1
        self.running = True
        self._trade_task = asyncio.create_task(asyncio.sleep(0.01))

    async def stop(self) -> None:
        self.stopped += 1
        self.running = False
        task = self._trade_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._trade_task = None


class _SupervisorAwareComponent:
    def __init__(self) -> None:
        self.received_supervisor: TaskSupervisor | None = None
        self.start_calls = 0
        self.stop_calls = 0
        self._task: asyncio.Task[Any] | None = None
        self._price_task: asyncio.Task[Any] | None = None

    def set_task_supervisor(self, supervisor: TaskSupervisor) -> None:
        self.received_supervisor = supervisor

    async def start(self, *, task_supervisor: TaskSupervisor) -> None:
        self.start_calls += 1
        self.received_supervisor = task_supervisor

        async def _sleep() -> None:
            await asyncio.sleep(0.05)

        self._task = task_supervisor.create(_sleep(), name="supervised-component-task")
        self._price_task = self._task

    async def stop(self) -> None:
        self.stop_calls += 1
        task = self._task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._task = None
        self._price_task = None


class _StubTradingManager:
    def __init__(self) -> None:
        self._risk_config = RiskConfig(max_total_exposure_pct=Decimal("0.5"))
        self._interface_metadata = {"metadata": {"source": "stub"}}

    def describe_risk_interface(self):
        return dict(self._interface_metadata)

    def get_risk_status(self):
        return {
            "risk_config": self._risk_config.dict(),
            "policy_limits": {"max_positions": 5},
            "policy_research_mode": False,
        }

    def get_last_risk_snapshot(self):
        return None

    def get_last_policy_snapshot(self):
        return None

    def get_last_roi_snapshot(self):
        return None


class _BrokenTradingManager:
    def describe_risk_interface(self):
        raise RiskApiError("broken interface", details={"manager": "broken"})

    def get_risk_status(self):
        return {"risk_config": "invalid"}

    def get_last_risk_snapshot(self):  # pragma: no cover - returning None is sufficient
        return None

    def get_last_policy_snapshot(self):  # pragma: no cover - returning None is sufficient
        return None

    def get_last_roi_snapshot(self):  # pragma: no cover - returning None is sufficient
        return None


class _StubPilot:
    def __init__(self, sensory: _StubSensory, broker: _StubBroker) -> None:
        self.sensory = sensory
        self.broker = broker
        self.started = 0
        self.stopped = 0
        self._snapshot = FixPilotSnapshot(
            status=FixPilotStatus.passed,
            timestamp=datetime.now(tz=UTC),
            components=(
                FixPilotComponent(name="sessions", status=FixPilotStatus.passed, details={}),
            ),
            metadata={},
        )

    async def start(self) -> None:
        self.started += 1
        self.sensory.running = True
        self.broker.running = True

    async def stop(self) -> None:
        self.stopped += 1
        self.sensory.running = False
        self.broker.running = False

    def snapshot(self) -> FixPilotSnapshot:
        return self._snapshot

    async def start(self) -> None:
        self.started += 1
        self.running = True
        self._trade_task = asyncio.create_task(asyncio.sleep(0.01))

    async def stop(self) -> None:
        self.stopped += 1
        self.running = False
        task = self._trade_task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._trade_task = None


class _StubBus:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, name, payload):
        self.events.append((name, payload))


class _StubMarketDataClient:
    def __init__(self) -> None:
        self.subscribed: list[tuple[tuple[str, ...], int]] = []
        self.unsubscribed: list[tuple[str, ...] | None] = []

    def subscribe_market_data(self, symbols, *, depth: int = 20) -> bool:
        self.subscribed.append((tuple(symbols), depth))
        return True

    def unsubscribe_market_data(self, symbols=None) -> bool:
        if symbols is None:
            self.unsubscribed.append(None)
        else:
            self.unsubscribed.append(tuple(symbols))
        return True


@pytest.mark.asyncio
async def test_professional_predator_app_lifecycle_tracks_components():
    config = SystemConfig()
    event_bus = EventBus()
    sensory = _StubSensory()
    broker = _StubBroker()
    fix_manager = _StubFixManager()

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=sensory,
        broker_interface=broker,
        fix_connection_manager=fix_manager,
        sensors={"stub": _StubSensor()},
    )
    app.add_cleanup_callback(fix_manager.stop_sessions)

    await app.start()

    assert sensory.started == 1
    assert broker.started == 1
    summary = app.summary()
    assert summary["status"] == "RUNNING"
    assert summary["components"]["sensory_running"] is True
    assert summary["components"]["broker_running"] is True
    queue_metrics = summary["components"].get("queue_metrics", {})
    assert queue_metrics["price"]["delivered"] == 5
    assert queue_metrics["trade"]["delivered"] == 7

    await app.shutdown()

    assert sensory.stopped == 1
    assert broker.stopped == 1
    assert fix_manager.stopped == 1
    stopped_summary = app.summary()
    assert stopped_summary["status"] == "STOPPED"
    assert stopped_summary["components"]["broker_running"] is False


@pytest.mark.asyncio
async def test_fix_sensory_organ_start_stop_awaits_worker():
    queue: asyncio.Queue = asyncio.Queue()
    supervisor = TaskSupervisor(namespace="test-sensory")

    def factory(coro, name=None):
        return supervisor.create(coro, name=name)

    organ = FIXSensoryOrgan(_StubBus(), queue, config={}, task_factory=factory)

    await organ.start()
    task = getattr(organ, "_price_task")
    assert isinstance(task, asyncio.Task)
    assert supervisor.active_count == 1

    await organ.stop()
    assert getattr(organ, "_price_task") is None
    assert organ.running is False
    assert supervisor.active_count == 0


@pytest.mark.asyncio
async def test_fix_sensory_organ_subscribes_and_unsubscribes_market_data():
    queue: asyncio.Queue = asyncio.Queue()
    client = _StubMarketDataClient()

    organ = FIXSensoryOrgan(
        _StubBus(),
        queue,
        config={"extras": {"FIX_MARKET_DATA_SYMBOLS": "1,2"}},
        market_data_client=client,
    )

    await organ.start()
    assert client.subscribed == [(('1', '2'), 20)]

    await organ.stop()
    assert client.unsubscribed == [("1", "2")]


@pytest.mark.asyncio
async def test_fix_broker_interface_start_stop_awaits_worker():
    queue: asyncio.Queue = asyncio.Queue()
    supervisor = TaskSupervisor(namespace="test-broker")

    def factory(coro, name=None):
        return supervisor.create(coro, name=name)

    broker = FIXBrokerInterface(_StubBus(), queue, fix_initiator=object(), task_factory=factory)

    await broker.start()
    task = getattr(broker, "_trade_task")
    assert isinstance(task, asyncio.Task)
    assert supervisor.active_count == 1

    await broker.stop()
    assert getattr(broker, "_trade_task") is None
    assert broker.running is False
    assert supervisor.active_count == 0


@pytest.mark.asyncio
async def test_professional_predator_app_passes_task_supervisor_to_components():
    config = SystemConfig()
    event_bus = EventBus()
    component = _SupervisorAwareComponent()
    supervisor = TaskSupervisor(namespace="test-supervised-component")

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=component,
        broker_interface=None,
        fix_connection_manager=None,
        sensors={"stub": _StubSensor()},
        task_supervisor=supervisor,
    )

    await app.start()
    await asyncio.sleep(0)

    assert component.start_calls == 1
    assert component.received_supervisor is supervisor

    snapshots = supervisor.describe()
    assert any(entry.get("name") == "supervised-component-task" for entry in snapshots)

    await app.shutdown()
    assert component.stop_calls == 1
    assert supervisor.active_count == 0


@pytest.mark.asyncio
async def test_professional_predator_app_with_fix_pilot_summary():
    config = SystemConfig()
    event_bus = EventBus()
    sensory = _StubSensory()
    broker = _StubBroker()
    fix_manager = _StubFixManager()

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=sensory,
        broker_interface=broker,
        fix_connection_manager=fix_manager,
        sensors={"stub": _StubSensor()},
    )
    pilot = _StubPilot(sensory, broker)
    app.attach_fix_pilot(pilot)

    await app.start()
    assert pilot.started == 1

    app.record_fix_pilot_snapshot(pilot.snapshot())
    summary = app.summary()
    assert summary["components"]["fix_pilot"]["status"] == FixPilotStatus.passed.value

    await app.shutdown()
    assert pilot.stopped == 1


@pytest.mark.asyncio
async def test_professional_predator_app_summary_includes_risk_interface():
    config = SystemConfig()
    event_bus = EventBus()
    sensory = _StubSensory()
    sensory.trading_manager = _StubTradingManager()
    broker = _StubBroker()

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=sensory,
        broker_interface=broker,
        fix_connection_manager=_StubFixManager(),
        sensors={"stub": _StubSensor()},
    )

    await app.start()
    summary = app.summary()
    risk_section = summary["risk"]
    assert risk_section["runbook"] == RISK_API_RUNBOOK
    runtime_metadata = risk_section["runtime"]
    assert runtime_metadata["runbook"] == RISK_API_RUNBOOK
    assert runtime_metadata["policy_limits"]["max_positions"] == 5

    interface_payload = risk_section["interface"]
    assert interface_payload["summary"]["max_total_exposure_pct"] == 0.5
    assert interface_payload["config"]["max_total_exposure_pct"] == 0.5
    assert interface_payload["metadata"]["source"] == "stub"

    await app.shutdown()


@pytest.mark.asyncio
async def test_professional_predator_app_summary_exposes_risk_api_errors():
    config = SystemConfig()
    event_bus = EventBus()
    sensory = _StubSensory()
    sensory.trading_manager = _BrokenTradingManager()
    broker = _StubBroker()

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=sensory,
        broker_interface=broker,
        fix_connection_manager=_StubFixManager(),
        sensors={"stub": _StubSensor()},
    )

    await app.start()
    summary = app.summary()
    risk_section = summary["risk"]
    errors = risk_section.get("errors") or {}
    assert "runtime" in errors
    assert errors["runtime"]["runbook"] == RISK_API_RUNBOOK
    runtime_details = errors["runtime"].get("details") or {}
    assert runtime_details.get("manager") == "_BrokenTradingManager"
    assert "interface" in errors
    interface_details = errors["interface"].get("details") or {}
    assert interface_details.get("manager") == "broken"

    await app.shutdown()


@pytest.mark.asyncio
async def test_professional_predator_app_summary_includes_event_bus_tasks():
    config = SystemConfig()
    event_bus = EventBus()
    sensory = _StubSensory()
    broker = _StubBroker()

    await event_bus.start()

    app = ProfessionalPredatorApp(
        config=config,
        event_bus=event_bus,
        sensory_organ=sensory,
        broker_interface=broker,
        fix_connection_manager=_StubFixManager(),
        sensors={"stub": _StubSensor()},
    )

    await app.start()
    try:
        summary = app.summary()
        task_entries = summary.get("event_bus_tasks")
        assert isinstance(task_entries, list)
        assert any(
            entry.get("metadata", {}).get("task") == "worker" for entry in task_entries
        )
    finally:
        await app.shutdown()
        await event_bus.stop()
