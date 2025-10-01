from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.core.event_bus import Event, EventBus, EventBusStatistics
from src.operations.event_bus_failover import EventPublishError
from src.operations.event_bus_health import (
    EventBusHealthStatus,
    evaluate_event_bus_health,
    publish_event_bus_health,
)


pytestmark = pytest.mark.guardrail


@pytest.mark.asyncio()
async def test_event_bus_health_reports_failure_when_bus_not_running() -> None:
    bus = EventBus()

    snapshot = evaluate_event_bus_health(bus, expected=True)

    assert snapshot.status is EventBusHealthStatus.fail
    assert snapshot.running is False
    assert "event bus" in snapshot.to_markdown().lower()


@pytest.mark.asyncio()
async def test_event_bus_health_ok_when_running() -> None:
    bus = EventBus()
    await bus.start()

    received: list[Event] = []

    async def _handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("demo", _handler)

    await bus.publish(Event(type="demo", payload={"value": 1}))
    await asyncio.sleep(0.05)

    snapshot = evaluate_event_bus_health(bus, expected=True)

    assert snapshot.running is True
    assert snapshot.status in {EventBusHealthStatus.ok, EventBusHealthStatus.warn}
    assert snapshot.published_events >= 1

    await bus.stop()


@pytest.mark.asyncio()
async def test_event_bus_health_warns_on_handler_errors() -> None:
    bus = EventBus()
    await bus.start()

    def _handler(event: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe("demo", _handler)

    await bus.publish(Event(type="demo", payload={"value": 2}))
    await asyncio.sleep(0.05)

    snapshot = evaluate_event_bus_health(bus, expected=True)

    assert snapshot.handler_errors >= 1
    assert snapshot.status in {EventBusHealthStatus.warn, EventBusHealthStatus.fail}

    await bus.stop()


@pytest.mark.asyncio()
async def test_publish_event_bus_health_dispatches_snapshot() -> None:
    bus = EventBus()
    await bus.start()

    received: list[Event] = []

    async def _handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("telemetry.event_bus.health", _handler)

    snapshot = evaluate_event_bus_health(bus, expected=True)
    publish_event_bus_health(bus, snapshot)
    await asyncio.sleep(0.05)

    assert received
    assert received[0].type == "telemetry.event_bus.health"

    await bus.stop()


def test_publish_event_bus_health_falls_back_to_global_bus(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _DummyTopicBus:
        def __init__(self) -> None:
            self.published: list[tuple[str, Any, str | None]] = []

        def publish_sync(self, topic: str, payload: Any, *, source: str | None = None) -> int:
            self.published.append((topic, payload, source))
            return 1

    captured_topic_bus = _DummyTopicBus()

    def _fake_get_global_bus() -> _DummyTopicBus:
        return captured_topic_bus

    monkeypatch.setattr(
        "src.operations.event_bus_health.get_global_bus", _fake_get_global_bus
    )

    bus = EventBus()

    def _failing_publish(_: Event) -> None:
        raise RuntimeError("bus offline")

    monkeypatch.setattr(bus, "publish_from_sync", _failing_publish)
    monkeypatch.setattr(bus, "is_running", lambda: True)

    snapshot = evaluate_event_bus_health(bus, expected=False)

    with caplog.at_level(logging.WARNING):
        publish_event_bus_health(bus, snapshot)

    assert captured_topic_bus.published
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "falling back to global bus" in messages


def test_publish_event_bus_health_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()

    def _unexpected(_: Event) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(bus, "publish_from_sync", _unexpected)
    monkeypatch.setattr(bus, "is_running", lambda: True)

    snapshot = evaluate_event_bus_health(bus, expected=False)

    with pytest.raises(EventPublishError) as excinfo:
        publish_event_bus_health(bus, snapshot)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == "telemetry.event_bus.health"


def test_event_bus_health_escalates_queue_backlog_and_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(tz=UTC)
    stats = EventBusStatistics(
        running=True,
        loop_running=True,
        queue_size=250,
        queue_capacity=500,
        subscriber_count=4,
        topic_subscribers={"telemetry": 3},
        published_events=12,
        dropped_events=6,
        handler_errors=4,
        last_event_timestamp=(now - timedelta(seconds=600)).timestamp(),
        last_error_timestamp=(now - timedelta(seconds=120)).timestamp(),
        started_at=(now - timedelta(hours=2)).timestamp(),
        uptime_seconds=7200.0,
    )

    bus = EventBus()
    monkeypatch.setattr(bus, "get_statistics", lambda: stats)

    snapshot = evaluate_event_bus_health(
        bus,
        expected=True,
        metadata={"region": "primary"},
        service="runtime", 
        queue_warn_threshold=100,
        queue_fail_threshold=200,
        idle_warn_seconds=300.0,
        now=now,
    )

    assert snapshot.status is EventBusHealthStatus.fail
    assert snapshot.metadata["expected"] is True
    assert snapshot.metadata["region"] == "primary"
    assert snapshot.service == "runtime"
    assert snapshot.queue_capacity == 500
    assert snapshot.last_event_at is not None
    assert snapshot.last_error_at is not None
    assert any("Queue has 250" in issue for issue in snapshot.issues)
    assert any("events dropped" in issue for issue in snapshot.issues)
    assert any("handler error" in issue for issue in snapshot.issues)
    assert any("No events observed" in issue for issue in snapshot.issues)


def test_publish_event_bus_health_raises_when_global_bus_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bus = EventBus()

    def _primary_failure(_: Event) -> None:
        raise RuntimeError("primary bus offline")

    monkeypatch.setattr(bus, "publish_from_sync", _primary_failure)
    monkeypatch.setattr(bus, "is_running", lambda: True)

    class _FailingGlobalBus:
        def publish_sync(
            self, topic: str, payload: object, *, source: str | None = None
        ) -> None:
            raise RuntimeError("global bus offline")

    failing_bus = _FailingGlobalBus()
    monkeypatch.setattr(
        "src.operations.event_bus_health.get_global_bus", lambda: failing_bus
    )

    snapshot = evaluate_event_bus_health(bus, expected=False)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EventPublishError) as excinfo:
            publish_event_bus_health(bus, snapshot)

    assert "Global event bus not running" in caplog.text
    assert excinfo.value.stage == "global"
    assert excinfo.value.event_type == "telemetry.event_bus.health"
