from datetime import UTC, datetime

from src.operations.alerts import AlertSeverity
from src.operations.incident_response import IncidentResponseSnapshot, IncidentResponseStatus
from src.operations.operational_readiness import (
    OperationalReadinessStatus,
    derive_operational_alerts,
    evaluate_operational_readiness,
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

    suppressed = derive_operational_alerts(
        readiness, threshold=OperationalReadinessStatus.fail, include_overall=False
    )
    assert suppressed == []
