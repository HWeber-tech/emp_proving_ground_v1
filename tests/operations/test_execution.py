from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Mapping

import pytest

from src.operations.event_bus_failover import EventPublishError
from src.operations.execution import (
    ExecutionPolicy,
    ExecutionReadinessSnapshot,
    ExecutionState,
    ExecutionStatus,
    evaluate_execution_readiness,
    format_execution_markdown,
    publish_execution_snapshot,
)


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> bool:
        self.events.append(event)
        return True

    def is_running(self) -> bool:
        return True


class _StubGlobalBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object], str]] = []

    def publish_sync(
        self, event_type: str, payload: Mapping[str, object], *, source: str
    ) -> None:
        self.events.append((event_type, dict(payload), source))


def _build_snapshot() -> ExecutionReadinessSnapshot:
    policy = ExecutionPolicy()
    state = ExecutionState(
        orders_submitted=5,
        orders_executed=5,
        connection_healthy=True,
        last_execution_at=datetime.now(timezone.utc),
    )
    return evaluate_execution_readiness(policy, state)


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


def test_publish_execution_snapshot_prefers_runtime_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _StubEventBus()

    def _unexpected_global() -> None:
        raise AssertionError("global bus should not be used when runtime succeeds")

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", _unexpected_global
    )

    snapshot = _build_snapshot()

    publish_execution_snapshot(bus, snapshot, source="test")

    assert bus.events
    event = bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.operational.execution"
    assert getattr(event, "source", "") == "test"


def test_publish_execution_snapshot_falls_back_on_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bus = _StubEventBus()

    def _failing_publish(_: object) -> bool:
        raise RuntimeError("primary bus offline")

    bus.publish_from_sync = _failing_publish  # type: ignore[assignment]

    global_bus = _StubGlobalBus()
    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", lambda: global_bus
    )

    snapshot = _build_snapshot()

    with caplog.at_level(logging.WARNING):
        publish_execution_snapshot(bus, snapshot, source="test")

    assert not bus.events
    assert global_bus.events
    event_type, payload, source = global_bus.events[-1]
    assert event_type == "telemetry.operational.execution"
    assert payload["status"] in {status.value for status in ExecutionStatus}
    assert source == "test"
    assert any(
        "falling back to global bus" in message.lower() for message in caplog.messages
    )


def test_publish_execution_snapshot_raises_on_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _StubEventBus()

    def _unexpected(_: object) -> bool:
        raise ValueError("boom")

    bus.publish_from_sync = _unexpected  # type: ignore[assignment]

    called: list[Any] = []

    def _global_bus() -> _StubGlobalBus:
        called.append("invoked")
        return _StubGlobalBus()

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", _global_bus)

    snapshot = _build_snapshot()

    with pytest.raises(EventPublishError) as exc:
        publish_execution_snapshot(bus, snapshot)

    assert exc.value.stage == "runtime"
    assert not called, "global bus should not be invoked when runtime raises"


def test_publish_execution_snapshot_raises_when_global_bus_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = _StubEventBus()

    def _runtime_failure(_: object) -> bool:
        raise RuntimeError("runtime offline")

    bus.publish_from_sync = _runtime_failure  # type: ignore[assignment]

    def _global_failure() -> None:
        raise RuntimeError("global offline")

    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", _global_failure)

    snapshot = _build_snapshot()

    with pytest.raises(EventPublishError) as exc:
        publish_execution_snapshot(bus, snapshot)

    assert exc.value.stage == "global"
