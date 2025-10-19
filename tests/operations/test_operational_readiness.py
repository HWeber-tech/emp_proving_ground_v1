import datetime as _datetime
import enum as _enum
import typing as _typing

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = _StrEnum


def _shim_class_getitem(name: str) -> type:
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _Placeholder.__name__ = name
    return _Placeholder


if not hasattr(_typing, "Unpack"):
    _typing.Unpack = _shim_class_getitem("Unpack")  # type: ignore[attr-defined]

if not hasattr(_typing, "NotRequired"):
    _typing.NotRequired = _shim_class_getitem("NotRequired")  # type: ignore[attr-defined]

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
from src.operations.incident_response import (
    IncidentResponsePolicy,
    IncidentResponseSnapshot,
    IncidentResponseState,
    IncidentResponseStatus,
    evaluate_incident_response,
)
from src.operations.sensory_drift import (
    DriftSeverity,
    SensoryDimensionDrift,
    SensoryDriftSnapshot,
)
from src.operations.drift_sentry import (
    DriftSentryConfig,
    DriftSentrySnapshot,
    evaluate_drift_sentry,
)
from src.operations.operational_readiness import (
    OperationalReadinessStatus,
    derive_operational_alerts,
    evaluate_operational_readiness,
    evaluate_operational_readiness_gate,
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


def _drift_snapshot(severity: DriftSeverity) -> SensoryDriftSnapshot:
    dimension = SensoryDimensionDrift(
        name="WHY",
        current_signal=0.8,
        baseline_signal=0.1,
        delta=0.7,
        current_confidence=0.75,
        baseline_confidence=0.65,
        confidence_delta=0.1,
        severity=severity,
        samples=12,
        page_hinkley_stat=1.2,
        variance_ratio=1.7,
        detectors=("page_hinkley_alert",),
    )
    return SensoryDriftSnapshot(
        generated_at=datetime(2025, 1, 3, tzinfo=UTC),
        status=severity,
        dimensions={"WHY": dimension},
        sample_window=12,
        metadata={"source": "test"},
    )


def _understanding_drift_snapshot(severity: DriftSeverity) -> DriftSentrySnapshot:
    if severity is DriftSeverity.alert:
        config = DriftSentryConfig(
            baseline_window=8,
            evaluation_window=4,
            min_observations=4,
            page_hinkley_delta=0.001,
            page_hinkley_warn=0.15,
            page_hinkley_alert=0.3,
            cusum_warn=1.2,
            cusum_alert=2.0,
            variance_ratio_warn=1.2,
            variance_ratio_alert=1.6,
        )
        series = [0.05] * 8 + [0.9] * 4
    elif severity is DriftSeverity.warn:
        config = DriftSentryConfig(
            baseline_window=8,
            evaluation_window=4,
            min_observations=4,
            page_hinkley_delta=0.002,
            page_hinkley_warn=0.08,
            page_hinkley_alert=0.35,
            cusum_warn=1.0,
            cusum_alert=1.8,
            variance_ratio_warn=1.25,
            variance_ratio_alert=3.0,
        )
        baseline = [0.12, 0.13, 0.11, 0.12, 0.13, 0.12, 0.11, 0.12]
        evaluation = [0.18, 0.2, 0.19, 0.18]
        series = baseline + evaluation
    else:
        config = DriftSentryConfig(
            baseline_window=8,
            evaluation_window=4,
            min_observations=4,
        )
        series = [0.2] * 12

    snapshot = evaluate_drift_sentry(
        {"belief_confidence": series},
        config=config,
        generated_at=datetime(2025, 1, 4, tzinfo=UTC),
    )
    assert snapshot.status is severity
    return snapshot


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


def test_operational_readiness_includes_drift_component() -> None:
    readiness = evaluate_operational_readiness(
        drift_snapshot=_drift_snapshot(DriftSeverity.warn)
    )

    assert readiness.status is OperationalReadinessStatus.warn
    component_statuses = readiness.metadata["component_statuses"]
    assert component_statuses == {"drift_sentry": "warn"}

    drift_component = readiness.components[0]
    assert drift_component.name == "drift_sentry"
    assert drift_component.summary.startswith("WHY:warn")
    assert "page_hinkley_alert" in drift_component.summary
    assert drift_component.metadata.get("issue_counts", {}).get("warn") == 1


def test_operational_readiness_includes_understanding_drift_component() -> None:
    readiness = evaluate_operational_readiness(
        drift_snapshot=_understanding_drift_snapshot(DriftSeverity.alert)
    )

    assert readiness.status is OperationalReadinessStatus.fail
    component = readiness.components[0]
    assert component.name == "drift_sentry"
    assert "belief_confidence:alert" in component.summary
    metadata = component.metadata
    assert metadata.get("runbook") == "docs/operations/runbooks/drift_sentry_response.md"
    issue_details = metadata.get("issue_details")
    assert issue_details and "belief_confidence" in issue_details
    assert issue_details["belief_confidence"].get("severity") == DriftSeverity.alert.value
    snapshot_metadata = metadata.get("snapshot", {}).get("metadata", {})
    theory_packet = snapshot_metadata.get("theory_packet")
    assert isinstance(theory_packet, dict)
    assert theory_packet.get("severity") == "alert"


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


def test_operational_readiness_alerts_include_drift_component() -> None:
    readiness = evaluate_operational_readiness(
        drift_snapshot=_drift_snapshot(DriftSeverity.alert)
    )

    events = derive_operational_alerts(readiness)
    categories = {event.category: event for event in events}
    assert "operational.drift_sentry" in categories
    drift_event = categories["operational.drift_sentry"]
    assert drift_event.severity is AlertSeverity.critical
    assert drift_event.tags == ("operational-readiness", "drift_sentry")
    assert drift_event.context["component"]["name"] == "drift_sentry"


def test_operational_readiness_alerts_include_understanding_drift_component() -> None:
    readiness = evaluate_operational_readiness(
        drift_snapshot=_understanding_drift_snapshot(DriftSeverity.warn)
    )

    events = derive_operational_alerts(readiness)
    drift_event = next(
        event for event in events if event.category == "operational.drift_sentry"
    )
    assert drift_event.severity is AlertSeverity.warning
    component_metadata = drift_event.context["component"]["metadata"]
    assert component_metadata.get("runbook") == "docs/operations/runbooks/drift_sentry_response.md"


def test_operational_readiness_aggregates_issue_counts() -> None:
    policy = IncidentResponsePolicy(required_runbooks=("redis_outage",))
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=None,
        drill_age_days=None,
        primary_oncall=tuple(),
        secondary_oncall=tuple(),
        open_incidents=tuple(),
        postmortem_backlog_hours=30.0,
        chatops_ready=False,
    )

    incident_snapshot = evaluate_incident_response(policy, state, service="emp_incidents")
    readiness = evaluate_operational_readiness(incident_response=incident_snapshot)

    issue_counts = readiness.metadata.get("issue_counts")
    assert issue_counts and issue_counts["fail"] >= 1
    component_details = readiness.metadata.get("component_issue_details")
    assert component_details and "incident_response" in component_details
    incident_details = component_details["incident_response"]
    assert "issue_counts" in incident_details


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
        include_gate_event=True,
    )

    assert len(results) == 3
    assert all(isinstance(result, AlertDispatchResult) for result in results)
    assert len(recording.events) == 3
    categories = {event.category for event in recording.events}
    assert categories == {
        "operational.readiness",
        "operational.system_validation",
        "operational.readiness.gate",
    }
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


def test_operational_readiness_gate_blocks_fail_components() -> None:
    readiness = evaluate_operational_readiness(
        incident_response=_incident_response_snapshot(IncidentResponseStatus.fail)
    )

    gate = evaluate_operational_readiness_gate(readiness)

    assert gate.should_block is True
    assert any("incident_response" in reason for reason in gate.blocking_reasons)
    assert gate.status is OperationalReadinessStatus.fail


def test_operational_readiness_gate_warn_components() -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.warn)
    )

    gate = evaluate_operational_readiness_gate(
        readiness,
        block_on_warn=False,
        warn_components=(),
        max_warn_components=1,
    )

    assert gate.should_block is False
    assert gate.status is OperationalReadinessStatus.warn
    assert gate.warnings

    blocking_gate = evaluate_operational_readiness_gate(
        readiness,
        block_on_warn=True,
        warn_components=("system_validation",),
        max_warn_components=0,
    )

    assert blocking_gate.should_block is True
    assert any("system_validation" in reason for reason in blocking_gate.blocking_reasons)


def test_operational_readiness_alerts_include_gate_event() -> None:
    readiness = evaluate_operational_readiness(
        system_validation=_system_validation_snapshot(SystemValidationStatus.warn)
    )
    gate = evaluate_operational_readiness_gate(readiness, block_on_warn=True)

    events = derive_operational_alerts(
        readiness,
        include_gate_event=True,
        gate_result=gate,
    )

    categories = {event.category for event in events}
    assert "operational.readiness.gate" in categories
    gate_event = next(event for event in events if event.category == "operational.readiness.gate")
    assert gate_event.context["gate"]["should_block"] is True
    assert gate_event.severity is AlertSeverity.warning
