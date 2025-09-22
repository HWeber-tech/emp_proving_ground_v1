from __future__ import annotations

import asyncio

import pytest

from src.core.event_bus import Event, EventBus
from src.operations.event_bus_health import (
    EventBusHealthStatus,
    evaluate_event_bus_health,
    publish_event_bus_health,
)


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
