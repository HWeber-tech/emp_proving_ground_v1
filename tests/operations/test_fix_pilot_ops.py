from datetime import UTC, datetime
from unittest.mock import patch

from src.core.event_bus import Event
from src.operations.fix_pilot import (
    FixPilotPolicy,
    FixPilotStatus,
    FixPilotSnapshot,
    evaluate_fix_pilot,
    format_fix_pilot_markdown,
    publish_fix_pilot_snapshot,
)
from src.runtime.fix_pilot import FixPilotState


def test_evaluate_fix_pilot_pass():
    state = FixPilotState(
        sessions_started=True,
        sensory_running=True,
        broker_running=True,
        queue_metrics={"price": {"delivered": 5, "dropped": 0}},
        active_orders=2,
        last_order={"order_id": "ORD-2", "status": "ACK"},
        compliance_summary={"policy": {"name": "default"}},
        risk_summary={"avg_latency_ms": 120.0},
        dropcopy_running=True,
        dropcopy_backlog=0,
        last_dropcopy_event=None,
        dropcopy_reconciliation=None,
        timestamp=datetime.now(tz=UTC),
    )
    policy = FixPilotPolicy()
    snapshot = evaluate_fix_pilot(policy, state, metadata={"ingest_success": True})

    assert snapshot.status is FixPilotStatus.passed
    components = {comp.name: comp for comp in snapshot.components}
    assert "broker" in components
    assert components["dropcopy"].status is FixPilotStatus.passed
    assert "orders" in components
    markdown = format_fix_pilot_markdown(snapshot)
    assert "FIX Pilot Status" in markdown


def test_evaluate_fix_pilot_warn_and_fail():
    state = FixPilotState(
        sessions_started=False,
        sensory_running=False,
        broker_running=True,
        queue_metrics={"trade": {"delivered": 0, "dropped": 3}},
        active_orders=0,
        last_order=None,
        compliance_summary=None,
        risk_summary=None,
        dropcopy_running=False,
        dropcopy_backlog=2,
        last_dropcopy_event=None,
        dropcopy_reconciliation={"status_mismatches": ["ORD-1"]},
        timestamp=datetime.now(tz=UTC),
    )
    policy = FixPilotPolicy(require_compliance=True, max_queue_drops=1)
    snapshot = evaluate_fix_pilot(policy, state)

    components = {component.name: component for component in snapshot.components}
    assert components["sessions"].status is FixPilotStatus.fail
    assert components["sensory"].status is FixPilotStatus.fail
    assert components["queues"].status is FixPilotStatus.warn
    assert components["compliance"].status is FixPilotStatus.warn
    assert components["dropcopy"].status is FixPilotStatus.warn
    assert "orders" in components


class _StubRuntimeBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1


class _StubTopicBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object], str | None]] = []

    def publish_sync(
        self, topic: str, payload: dict[str, object], *, source: str | None = None
    ) -> int:
        self.published.append((topic, payload, source))
        return 1


def _snapshot() -> FixPilotSnapshot:
    return FixPilotSnapshot(
        status=FixPilotStatus.passed,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        components=(),
        metadata={},
    )


def test_publish_fix_pilot_snapshot_uses_failover_helper() -> None:
    runtime_bus = _StubRuntimeBus()
    snapshot = _snapshot()

    with patch("src.operations.fix_pilot.publish_event_with_failover") as helper:
        publish_fix_pilot_snapshot(
            runtime_bus,
            snapshot,
            channel="telemetry.execution.fix_pilot",
            source="unit.test.fix_pilot",
        )

    helper.assert_called_once()
    args, kwargs = helper.call_args
    assert args[0] is runtime_bus
    event = args[1]
    assert event.type == "telemetry.execution.fix_pilot"
    assert event.payload == snapshot.as_dict()
    assert event.source == "unit.test.fix_pilot"
    assert "logger" in kwargs and kwargs["logger"].name == "src.operations.fix_pilot"


def test_publish_fix_pilot_snapshot_without_runtime_bus_uses_factory() -> None:
    topic_bus = _StubTopicBus()
    snapshot = _snapshot()

    publish_fix_pilot_snapshot(
        None,
        snapshot,
        channel="telemetry.execution.fix_pilot",
        source="unit.test.fix_pilot",
        global_bus_factory=lambda: topic_bus,
    )

    assert topic_bus.published == [
        (
            "telemetry.execution.fix_pilot",
            snapshot.as_dict(),
            "unit.test.fix_pilot",
        )
    ]


def test_publish_fix_pilot_snapshot_direct_topic_bus() -> None:
    topic_bus = _StubTopicBus()
    snapshot = _snapshot()

    publish_fix_pilot_snapshot(
        topic_bus,
        snapshot,
        channel="telemetry.execution.fix_pilot",
        source="unit.test.fix_pilot",
    )

    assert topic_bus.published == [
        (
            "telemetry.execution.fix_pilot",
            snapshot.as_dict(),
            "unit.test.fix_pilot",
        )
    ]
