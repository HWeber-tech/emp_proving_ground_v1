from datetime import UTC, datetime

import pytest

from src.operations.alerts import (
    AlertChannel,
    AlertDispatchResult,
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from src.operations.event_bus_failover import EventPublishError
from src.operations.incident_response import IncidentResponseSnapshot, IncidentResponseStatus
from src.operations.operational_readiness import (
    OperationalReadinessStatus,
    derive_operational_alerts,
    evaluate_operational_readiness,
    publish_operational_readiness_snapshot,
    route_operational_readiness_alerts,
)
from src.operations.system_validation import (
    SystemValidationCheck,
    SystemValidationSnapshot,
    SystemValidationStatus,
)


def _system_validation_snapshot(status: SystemValidationStatus) -> SystemValidationSnapshot:
    return SystemValidationSnapshot(
        status=status,
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        total_checks=5,
        passed_checks=4,
        failed_checks=1,
        success_rate=0.8,
        checks=(
            SystemValidationCheck(name="database", passed=False, message="connection timeout"),
            SystemValidationCheck(name="event_bus", passed=True),
        ),
        metadata={"validator": "ops_guardian"},
    )


def _incident_response_snapshot(status: IncidentResponseStatus) -> IncidentResponseSnapshot:
    return IncidentResponseSnapshot(
        service="emp_incidents",
        generated_at=datetime(2025, 1, 2, tzinfo=UTC),
        status=status,
        missing_runbooks=("redis_outage",),
        training_age_days=120.0,
        drill_age_days=30.0,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("INC-42",),
        issues=("Postmortem backlog exceeds the configured SLA",),
        metadata={"postmortem_backlog_hours": 52.0},
    )


def test_operational_readiness_combines_components() -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.warn),
        incident_response=_incident_response_snapshot(IncidentResponseStatus.fail),
        metadata={"environment": "staging"},
    )

    assert readiness.status is OperationalReadinessStatus.fail
    assert readiness.metadata["component_count"] == 2
    assert readiness.metadata["status_breakdown"] == {"warn": 1, "fail": 1}
    assert readiness.metadata["component_statuses"] == {
        "system_validation": "warn",
        "incident_response": "fail",
    }
    names = {component.name for component in readiness.components}
    assert names == {"system_validation", "incident_response"}

    system_component = next(component for component in readiness.components if component.name == "system_validation")
    assert "failing checks" in system_component.summary
    assert "database" in system_component.summary

    markdown = readiness.to_markdown().lower()
    assert "operational readiness" in markdown
    assert "incident_response" in markdown


def test_operational_readiness_alert_generation() -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.warn)
    )

    events = derive_operational_alerts(readiness)

    assert len(events) == 2  # overall snapshot + component alert
    categories = {event.category: event for event in events}
    overall = categories["operational.readiness"]
    component = categories["operational.system_validation"]

    assert overall.severity is AlertSeverity.warning
    assert component.severity is AlertSeverity.warning
    assert "status warn" in component.message
    assert overall.context["snapshot"]["metadata"]["status_breakdown"] == {"warn": 1}
    assert overall.tags == ("operational-readiness",)
    assert component.tags == ("operational-readiness", "system_validation")

    suppressed = derive_operational_alerts(
        readiness, threshold=OperationalReadinessStatus.fail, include_overall=False
    )
    assert suppressed == []


class RecordingTransport:
    def __init__(self) -> None:
        self.events: list[object] = []

    def __call__(self, event: object) -> None:
        self.events.append(event)


class StubBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:  # pragma: no cover - simple stub
        self.events.append(event)

    def is_running(self) -> bool:  # pragma: no cover - simple stub
        return True


class RaisingRuntimeBus(StubBus):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def publish_from_sync(self, event: object) -> None:  # type: ignore[override]
        raise self._exc


class StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, str | None]] = []

    def publish_sync(self, topic: str, payload: object, *, source: str | None = None) -> None:
        self.events.append((topic, payload, source))


def _alert_manager(recording: RecordingTransport) -> AlertManager:
    channel = AlertChannel(
        name="recording",
        transport=recording,
        channel_type="memory",
        min_severity=AlertSeverity.info,
    )
    rule = AlertRule(
        name="all",
        categories=(),
        min_severity=AlertSeverity.info,
        channels=("recording",),
    )
    return AlertManager(channels=[channel], rules=[rule])


def test_route_operational_readiness_alerts_dispatches_events() -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.warn)
    )

    recording = RecordingTransport()
    manager = _alert_manager(recording)

    results = route_operational_readiness_alerts(
        manager,
        readiness,
        base_tags=("ops",),
    )

    assert len(results) == 2
    assert all(isinstance(result, AlertDispatchResult) for result in results)
    assert len(recording.events) == 2
    categories = {event.category for event in recording.events}
    assert categories == {"operational.readiness", "operational.system_validation"}
    for event in recording.events:
        assert event.tags[0] == "ops"


def test_publish_operational_readiness_snapshot_uses_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.passed)
    )

    runtime_bus = RaisingRuntimeBus(RuntimeError("loop stopped"))
    global_bus = StubTopicBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    publish_operational_readiness_snapshot(runtime_bus, readiness, source="test")

    assert not runtime_bus.events
    assert global_bus.events
    topic, payload, source = global_bus.events[-1]
    assert topic == "telemetry.operational.operational_readiness"
    assert payload.get("status") == OperationalReadinessStatus.ok.value
    assert source == "test"


def test_publish_operational_readiness_snapshot_raises_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.passed)
    )

    runtime_bus = RaisingRuntimeBus(ValueError("boom"))
    global_bus = StubTopicBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: global_bus)

    with pytest.raises(EventPublishError) as exc_info:
        publish_operational_readiness_snapshot(runtime_bus, readiness, source="test")

    assert exc_info.value.stage == "runtime"
    assert not runtime_bus.events
    assert not global_bus.events
