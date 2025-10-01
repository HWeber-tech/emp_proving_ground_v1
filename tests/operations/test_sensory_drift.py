from datetime import datetime

import pytest

from collections.abc import Callable
from typing import Any

from src.core.event_bus import Event
from src.operations.sensory_drift import (
    DriftSeverity,
    SensoryDriftSnapshot,
    evaluate_sensory_drift,
    publish_sensory_drift,
)


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


def test_evaluate_sensory_drift_flags_alert_and_warn() -> None:
    audit_entries = [
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.62,
            "confidence": 0.78,
            "dimensions": {
                "why": {"signal": 0.55, "confidence": 0.74},
                "how": {"signal": -0.15, "confidence": 0.68},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.10,
            "confidence": 0.45,
            "dimensions": {
                "why": {"signal": 0.05, "confidence": 0.40},
                "how": {"signal": 0.12, "confidence": 0.52},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": -0.05,
            "confidence": 0.48,
            "dimensions": {
                "why": {"signal": -0.02, "confidence": 0.42},
                "how": {"signal": 0.08, "confidence": 0.46},
            },
        },
    ]

    snapshot = evaluate_sensory_drift(audit_entries, metadata={"ingest_success": True})

    assert snapshot.status is DriftSeverity.alert
    assert snapshot.metadata["ingest_success"] is True
    assert snapshot.metadata["entries"] == len(audit_entries)

    why = snapshot.dimensions["why"]
    assert why.severity is DriftSeverity.alert
    assert why.baseline_signal is not None
    assert pytest.approx(why.baseline_signal, rel=1e-6) == 0.015
    assert why.delta is not None and why.delta > 0.5

    how = snapshot.dimensions["how"]
    assert how.severity is DriftSeverity.warn
    assert how.delta is not None and pytest.approx(how.delta, rel=1e-6) == -0.25
    markdown = snapshot.to_markdown()
    assert "why" in markdown and "how" in markdown


def test_evaluate_sensory_drift_handles_single_entry() -> None:
    audit_entries = [
        {
            "symbol": "GBPUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {
                "why": {"signal": 0.25, "confidence": 0.6},
            },
        }
    ]

    snapshot = evaluate_sensory_drift(audit_entries)

    assert snapshot.status is DriftSeverity.normal
    assert snapshot.sample_window == 1
    why = snapshot.dimensions["why"]
    assert why.baseline_signal is None
    assert why.delta is None
    assert why.severity is DriftSeverity.normal


def _snapshot() -> SensoryDriftSnapshot:
    entries = [
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {"why": {"signal": 0.6, "confidence": 0.8}},
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {"why": {"signal": 0.5, "confidence": 0.75}},
        },
    ]
    return evaluate_sensory_drift(entries)


def test_publish_sensory_drift_prefers_runtime_bus() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    snapshot = _snapshot()

    publish_sensory_drift(bus, snapshot, global_bus_factory=lambda: topic_bus)

    assert len(bus.events) == 1
    assert topic_bus.events == []
    event = bus.events[0]
    assert event.type == "telemetry.sensory.drift"
    assert event.source == "operations.sensory_drift"
    assert event.payload["status"] == snapshot.status.value


def test_publish_sensory_drift_falls_back_to_global_bus_on_none_result() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    def _none(event: Event) -> None:
        bus.events.append(event)
        return None

    bus.publish_from_sync = _none  # type: ignore[method-assign]

    snapshot = _snapshot()

    publish_sensory_drift(bus, snapshot, global_bus_factory=lambda: topic_bus)

    assert topic_bus.events
    event_type, payload, source = topic_bus.events[-1]
    assert event_type == "telemetry.sensory.drift"
    assert source == "operations.sensory_drift"
    assert payload["status"] == snapshot.status.value
