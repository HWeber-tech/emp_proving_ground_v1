from datetime import datetime, timezone

import pytest

from src.operations.execution import (
    ExecutionPolicy,
    ExecutionState,
    ExecutionStatus,
    evaluate_execution_readiness,
    format_execution_markdown,
    publish_execution_snapshot,
)


class StubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return True


class StubFallbackBus(StubBus):
    def publish_from_sync(self, event: object) -> None:  # type: ignore[override]
        super().publish_from_sync(event)
        return None


class StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[object, object, object | None]] = []

    def publish_sync(self, event_type: object, payload: object, *, source: object | None = None) -> None:
        self.events.append((event_type, payload, source))


def test_evaluate_execution_readiness_pass() -> None:
    policy = ExecutionPolicy()
    state = ExecutionState(
        orders_submitted=10,
        orders_executed=10,
        pending_orders=0,
        avg_latency_ms=120.0,
        max_latency_ms=240.0,
        drop_copy_active=True,
        connection_healthy=True,
        last_execution_at=datetime.now(timezone.utc),
    )

    snapshot = evaluate_execution_readiness(
        policy,
        state,
        metadata={"window": "intraday"},
        service="paper-execution",
    )

    assert snapshot.status is ExecutionStatus.passed
    assert snapshot.fill_rate == pytest.approx(1.0)
    markdown = format_execution_markdown(snapshot)
    assert "paper-execution" in markdown
    assert "STATUS" in markdown.upper()


def test_evaluate_execution_readiness_flags_failures() -> None:
    policy = ExecutionPolicy(
        min_fill_rate=0.9,
        max_rejection_rate=0.05,
        max_pending_orders=2,
        max_avg_latency_ms=150.0,
        require_connection=True,
        require_drop_copy=True,
        max_drop_copy_lag_seconds=5.0,
    )
    state = ExecutionState(
        orders_submitted=12,
        orders_executed=8,
        orders_failed=4,
        pending_orders=5,
        avg_latency_ms=320.0,
        max_latency_ms=900.0,
        drop_copy_lag_seconds=15.0,
        drop_copy_active=False,
        connection_healthy=False,
        drop_copy_metrics=(("dropped", 3),),
    )

    snapshot = evaluate_execution_readiness(policy, state, service="institutional")

    assert snapshot.status is ExecutionStatus.fail
    issue_codes = {issue.code for issue in snapshot.issues}
    assert "connection_down" in issue_codes
    assert "drop_copy_inactive" in issue_codes
    assert "pending_orders_exceeded" in issue_codes


def test_publish_execution_snapshot_emits_event() -> None:
    policy = ExecutionPolicy()
    state = ExecutionState(orders_submitted=5, orders_executed=5, connection_healthy=True)
    snapshot = evaluate_execution_readiness(policy, state)

    bus = StubBus()
    publish_execution_snapshot(bus, snapshot, source="test")
    assert bus.events, "expected execution telemetry event"
    event = bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.operational.execution"
    assert getattr(event, "source", "") == "test"
    payload = getattr(event, "payload", {})
    assert payload.get("status") in {status.value for status in ExecutionStatus}


def test_publish_execution_snapshot_falls_back_to_global_bus() -> None:
    policy = ExecutionPolicy()
    state = ExecutionState(orders_submitted=3, orders_executed=2, connection_healthy=True)
    snapshot = evaluate_execution_readiness(policy, state)

    bus = StubFallbackBus()
    topic_bus = StubTopicBus()

    publish_execution_snapshot(
        bus,
        snapshot,
        source="fallback",
        global_bus_factory=lambda: topic_bus,
    )

    assert topic_bus.events, "expected fallback global bus publish"
    event_type, payload, source = topic_bus.events[-1]
    assert event_type == "telemetry.operational.execution"
    assert source == "fallback"
    assert payload.get("status") in {status.value for status in ExecutionStatus}
