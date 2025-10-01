import logging
from collections.abc import Callable
from typing import Any

import pytest

from src.core.event_bus import Event
from src.operations.event_bus_failover import EventPublishError, publish_event_with_failover


class _StubEventBus:
    def __init__(self, *, running: bool = True) -> None:
        self.events: list[Event] = []
        self._running = running
        self.publish_from_sync: Callable[[Event], Any] | None = self._publish  # type: ignore[assignment]

    def _publish(self, event: Event) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return self._running


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str | None]] = []

    def publish_sync(self, event_type: str, payload: Any, *, source: str | None = None) -> None:
        self.events.append((event_type, payload, source))


def _event() -> Event:
    return Event(type="telemetry.test", payload={"ok": True}, source="test")


def test_publish_event_with_failover_prefers_runtime_bus() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    publish_event_with_failover(
        bus,
        _event(),
        logger=logging.getLogger("test"),
        runtime_fallback_message="runtime fallback",
        runtime_unexpected_message="runtime unexpected",
        runtime_none_message="runtime none",
        global_not_running_message="global missing",
        global_unexpected_message="global unexpected",
        global_bus_factory=lambda: topic_bus,
    )

    assert len(bus.events) == 1
    assert topic_bus.events == []


def test_publish_event_with_failover_falls_back_on_runtime_none() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    def _none(_: Event) -> None:
        return None

    bus.publish_from_sync = _none  # type: ignore[method-assign]

    publish_event_with_failover(
        bus,
        _event(),
        logger=logging.getLogger("test"),
        runtime_fallback_message="runtime fallback",
        runtime_unexpected_message="runtime unexpected",
        runtime_none_message="runtime none",
        global_not_running_message="global missing",
        global_unexpected_message="global unexpected",
        global_bus_factory=lambda: topic_bus,
    )

    assert topic_bus.events


def test_publish_event_with_failover_raises_on_unexpected_runtime_error() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    def _boom(_: Event) -> None:
        raise ValueError("boom")

    bus.publish_from_sync = _boom  # type: ignore[method-assign]

    with pytest.raises(EventPublishError) as exc_info:
        publish_event_with_failover(
            bus,
            _event(),
            logger=logging.getLogger("test"),
            runtime_fallback_message="runtime fallback",
            runtime_unexpected_message="runtime unexpected",
            runtime_none_message="runtime none",
            global_not_running_message="global missing",
            global_unexpected_message="global unexpected",
            global_bus_factory=lambda: topic_bus,
        )

    assert exc_info.value.stage == "runtime"
    assert topic_bus.events == []


def test_publish_event_with_failover_raises_on_global_error() -> None:
    bus = _StubEventBus(running=False)

    class _FailingTopicBus(_StubTopicBus):
        def publish_sync(self, event_type: str, payload: Any, *, source: str | None = None) -> None:  # type: ignore[override]
            raise RuntimeError("offline")

    with pytest.raises(EventPublishError) as exc_info:
        publish_event_with_failover(
            bus,
            _event(),
            logger=logging.getLogger("test"),
            runtime_fallback_message="runtime fallback",
            runtime_unexpected_message="runtime unexpected",
            runtime_none_message="runtime none",
            global_not_running_message="global missing",
            global_unexpected_message="global unexpected",
            global_bus_factory=_FailingTopicBus,
        )

    assert exc_info.value.stage == "global"
