import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime

import pytest

from src.operations.fix_pilot import FixPilotPolicy, FixPilotStatus, evaluate_fix_pilot
from src.runtime.fix_dropcopy import FixDropcopyReconciler
from src.runtime.fix_pilot import FixIntegrationPilot, FixPilotState
from src.runtime.task_supervisor import TaskSupervisor


class _StubAdapter:
    def __init__(self) -> None:
        self.queue = None
        self.metrics = {"delivered": 3, "dropped": 0}

    def set_message_queue(self, queue) -> None:
        self.queue = queue

    def get_queue_metrics(self):
        return dict(self.metrics)


class _StubManager:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.price = _StubAdapter()
        self.trade = _StubAdapter()
        self.dropcopy = _StubAdapter()
        self.initiator = object()

    def start_sessions(self) -> bool:
        self.started += 1
        return True

    def stop_sessions(self) -> None:
        self.stopped += 1

    def get_application(self, session: str):
        if session == "price":
            return self.price
        if session == "trade":
            return self.trade
        if session == "dropcopy":
            return self.dropcopy
        return None

    def get_initiator(self, session: str):
        if session == "trade":
            return self.initiator
        return None


class _StubComponent:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.running = False
        self._task: asyncio.Task | None = None
        self.queue = asyncio.Queue()

    async def start(self) -> None:
        self.started += 1
        self.running = True
        self._task = asyncio.create_task(asyncio.sleep(0))

    async def stop(self) -> None:
        self.stopped += 1
        self.running = False
        task = self._task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._task = None


class _StubBroker(_StubComponent):
    def __init__(self) -> None:
        super().__init__()
        self.trade_queue = self.queue
        self.fix_initiator = None
        self.orders = {"ORD-1": {"status": "ACK"}}

    def get_all_orders(self):
        return dict(self.orders)


class _StubComplianceMonitor:
    def summary(self):  # pragma: no cover - simple mapping
        return {"policy": {"name": "default"}}


@pytest.mark.asyncio
async def test_fix_integration_pilot_start_stop_and_snapshot():
    manager = _StubManager()
    sensory = _StubComponent()
    broker = _StubBroker()
    supervisor = TaskSupervisor(namespace="test", logger=logging.getLogger(__name__))

    def factory(coro, name=None):
        return supervisor.create(coro, name=name)

    dropcopy = FixDropcopyReconciler(
        event_bus=None,
        broker_order_lookup=broker.get_all_orders,
        task_factory=factory,
    )
    pilot = FixIntegrationPilot(
        connection_manager=manager,
        sensory_organ=sensory,
        broker_interface=broker,
        task_supervisor=supervisor,
        event_bus=None,
        compliance_monitor=_StubComplianceMonitor(),
        dropcopy_listener=dropcopy,
    )

    await pilot.start()
    assert supervisor.active_count >= 1
    assert manager.started == 1
    assert sensory.started == 1
    assert broker.started == 1
    assert broker.fix_initiator is manager.initiator
    assert manager.dropcopy.queue is dropcopy.dropcopy_queue

    await dropcopy.dropcopy_queue.put({11: b"ORD-1", 150: b"0", 39: b"ACK"})
    await asyncio.sleep(0)

    state = pilot.snapshot()
    assert state.sessions_started is True
    assert state.sensory_running is True
    assert state.broker_running is True
    assert state.queue_metrics["price"]["delivered"] == 3
    assert state.active_orders == 1
    assert state.compliance_summary["policy"]["name"] == "default"
    assert state.dropcopy_running is True
    assert state.dropcopy_backlog == 0
    assert state.dropcopy_reconciliation is not None

    await pilot.stop()
    assert manager.stopped == 1
    assert sensory.stopped == 1
    assert broker.stopped == 1
    assert supervisor.active_count == 0


def test_evaluate_fix_pilot_status():
    timestamp = datetime.now(tz=UTC)
    state = FixPilotState(
        sessions_started=True,
        sensory_running=True,
        broker_running=False,
        queue_metrics={"trade": {"delivered": 0, "dropped": 2}},
        active_orders=0,
        last_order=None,
        compliance_summary=None,
        risk_summary=None,
        dropcopy_running=False,
        dropcopy_backlog=5,
        last_dropcopy_event=None,
        dropcopy_reconciliation={"orders_without_dropcopy": ["ORD-1"]},
        timestamp=timestamp,
    )
    policy = FixPilotPolicy(max_queue_drops=0, require_broker=True)
    snapshot = evaluate_fix_pilot(policy, state)
    assert snapshot.status == FixPilotStatus.fail
    component_names = {comp.name: comp for comp in snapshot.components}
    assert component_names["broker"].status == FixPilotStatus.fail
    assert component_names["queues"].status == FixPilotStatus.warn
