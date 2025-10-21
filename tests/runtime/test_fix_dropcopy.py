import asyncio

import pytest

from src.runtime.task_supervisor import TaskSupervisor

from src.runtime.fix_dropcopy import FixDropcopyReconciler


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit_nowait(self, event) -> None:
        self.events.append(event.payload)


@pytest.mark.asyncio
async def test_dropcopy_reconciler_processes_messages_and_reconciles():
    bus = _StubEventBus()

    orders = {"ORD-1": {"status": "ACKNOWLEDGED"}}

    reconciler = FixDropcopyReconciler(
        event_bus=bus,
        broker_order_lookup=lambda: orders,
    )

    await reconciler.start()
    await reconciler.dropcopy_queue.put({11: b"ORD-1", 150: b"0", 39: b"ACK"})
    await asyncio.sleep(0)

    assert reconciler.get_backlog() == 0
    last_event = reconciler.get_last_event()
    assert last_event is not None
    assert last_event["order_id"] == "ORD-1"

    summary = reconciler.reconciliation_summary()
    assert summary["observed_orders"] == 1
    assert summary.get("orders_without_dropcopy", []) == []
    assert bus.events

    await reconciler.stop()


@pytest.mark.asyncio
async def test_dropcopy_reconciler_uses_task_factory():
    supervisor = TaskSupervisor(namespace="test-dropcopy")

    def factory(coro, name=None):
        return supervisor.create(coro, name=name)

    reconciler = FixDropcopyReconciler(
        event_bus=None,
        broker_order_lookup=lambda: {},
        task_factory=factory,
    )

    await reconciler.start()
    assert supervisor.active_count == 1

    await reconciler.stop()
    assert supervisor.active_count == 0


@pytest.mark.asyncio
async def test_dropcopy_reconciler_flags_unmatched_orders():
    reconciler = FixDropcopyReconciler(
        event_bus=None,
        broker_order_lookup=lambda: {"ORD-2": {"status": "NEW"}},
    )

    await reconciler.start()
    await reconciler.dropcopy_queue.put({11: b"ORD-1", 150: b"0"})
    await asyncio.sleep(0)

    summary = reconciler.reconciliation_summary()
    assert "orders_without_dropcopy" in summary
    assert summary["orders_without_dropcopy"] == ["ORD-2"]
    assert "status_mismatches" not in summary or not summary["status_mismatches"]

    await reconciler.stop()


def test_dropcopy_reconciler_parses_localised_floats():
    reconciler = FixDropcopyReconciler(event_bus=None, broker_order_lookup=None)

    event = reconciler._normalise_message({11: "ORD-3", 31: "1,234.5", 150: "0"})

    assert event is not None
    assert event.last_px == pytest.approx(1234.5)
