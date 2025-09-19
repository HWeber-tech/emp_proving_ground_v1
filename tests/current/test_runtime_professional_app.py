import asyncio
from contextlib import suppress

import pytest

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.runtime.predator_app import ProfessionalPredatorApp
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


class _StubBus:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, name, payload):
        self.events.append((name, payload))


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
    organ = FIXSensoryOrgan(_StubBus(), queue, config={})

    await organ.start()
    task = getattr(organ, "_price_task")
    assert isinstance(task, asyncio.Task)

    await organ.stop()
    assert getattr(organ, "_price_task") is None
    assert organ.running is False


@pytest.mark.asyncio
async def test_fix_broker_interface_start_stop_awaits_worker():
    queue: asyncio.Queue = asyncio.Queue()
    broker = FIXBrokerInterface(_StubBus(), queue, fix_initiator=object())

    await broker.start()
    task = getattr(broker, "_trade_task")
    assert isinstance(task, asyncio.Task)

    await broker.stop()
    assert getattr(broker, "_trade_task") is None
    assert broker.running is False
