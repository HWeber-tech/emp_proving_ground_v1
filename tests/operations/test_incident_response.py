from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.core.event_bus import Event
from src.operations.alerts import AlertDispatchResult, AlertEvent, AlertSeverity
from src.operations.incident_response import (
    IncidentResponseGateResult,
    IncidentResponseMetrics,
    IncidentResponsePolicy,
    IncidentResponseSnapshot,
    IncidentResponseState,
    IncidentResponseStatus,
    derive_incident_response_alerts,
    evaluate_incident_response,
    evaluate_incident_response_gate,
    publish_incident_response_snapshot,
    route_incident_response_alerts,
)
from src.operations.event_bus_failover import EventPublishError


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._running = True

    def is_running(self) -> bool:  # pragma: no cover - trivial proxy
        return self._running

    def publish_from_sync(self, event: Event) -> None:
        self.events.append(event)
        return True


class _StubAlertManager:
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    def dispatch(self, event: AlertEvent) -> AlertDispatchResult:
        self.events.append(event)
        return AlertDispatchResult(event=event, triggered_channels=("stub",))


def test_evaluate_incident_response_ok() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("redis_outage", "kafka_lag"),
        training_interval_days=30,
        drill_interval_days=45,
        minimum_primary_responders=2,
        minimum_secondary_responders=1,
        postmortem_sla_hours=24.0,
        maximum_open_incidents=2,
        require_chatops=True,
        require_major_incident_history=True,
        major_incident_review_interval_days=180,
    )
    state = IncidentResponseState(
        available_runbooks=("redis_outage", "kafka_lag", "fix_restart"),
        training_age_days=12.0,
        drill_age_days=20.0,
        primary_oncall=("alice", "bob"),
        secondary_oncall=("carol",),
        open_incidents=tuple(),
        postmortem_backlog_hours=8.0,
        chatops_ready=True,
        last_major_incident_age_days=45.0,
    )

    now = datetime(2025, 2, 1, tzinfo=UTC)
    snapshot = evaluate_incident_response(policy, state, now=now, service="emp")

    assert snapshot.status is IncidentResponseStatus.ok
    assert not snapshot.issues
    markdown = snapshot.to_markdown()
    assert "primary responders: 2" in markdown.lower()
    payload = snapshot.as_dict()
    assert payload["status"] == "ok"
    assert payload["metadata"]["policy"]["required_runbooks"] == [
        "redis_outage",
        "kafka_lag",
    ]


def test_evaluate_incident_response_escalates() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("redis_outage",),
        training_interval_days=14,
        drill_interval_days=14,
        minimum_primary_responders=1,
        minimum_secondary_responders=1,
        postmortem_sla_hours=12.0,
        maximum_open_incidents=0,
        require_chatops=True,
    )
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=40.0,
        drill_age_days=50.0,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("ingest-lag",),
        postmortem_backlog_hours=30.0,
        chatops_ready=False,
    )

    snapshot = evaluate_incident_response(policy, state, service="emp")

    assert snapshot.status is IncidentResponseStatus.fail
    assert any("missing" in issue.lower() for issue in snapshot.issues)
    assert any("postmortem" in issue.lower() for issue in snapshot.issues)
    assert snapshot.metadata["postmortem_backlog_hours"] == 30.0
    issue_counts = snapshot.metadata.get("issue_counts")
    assert issue_counts and issue_counts["fail"] >= 1
    assert snapshot.metadata.get("highest_issue_severity") == "fail"
    issue_details = snapshot.metadata.get("issue_details")
    assert issue_details
    by_message = {detail["message"]: detail for detail in issue_details}  # type: ignore[index]
    assert any("Missing required runbooks" in message for message in by_message)
    missing_detail = next(
        detail for message, detail in by_message.items() if "Missing required runbooks" in message
    )
    assert missing_detail["category"] == "missing_runbooks"
    assert missing_detail["severity"] == "fail"


def test_incident_response_mapping_parsing() -> None:
    mapping = {
        "INCIDENT_REQUIRED_RUNBOOKS": "redis_outage,kafka_lag",
        "INCIDENT_AVAILABLE_RUNBOOKS": ["redis_outage"],
        "INCIDENT_MIN_PRIMARY_RESPONDERS": "2",
        "INCIDENT_PRIMARY_RESPONDERS": "alice,bob",
        "INCIDENT_TRAINING_INTERVAL_DAYS": "20",
        "INCIDENT_TRAINING_AGE_DAYS": "25",
        "INCIDENT_DRILL_INTERVAL_DAYS": "30",
        "INCIDENT_DRILL_AGE_DAYS": "35",
        "INCIDENT_POSTMORTEM_SLA_HOURS": "18",
        "INCIDENT_POSTMORTEM_BACKLOG_HOURS": "8",
        "INCIDENT_REQUIRE_CHATOPS": "false",
        "INCIDENT_REQUIRE_MAJOR_HISTORY": "true",
        "INCIDENT_MAJOR_REVIEW_INTERVAL_DAYS": "200",
        "INCIDENT_MAJOR_REVIEW_FAIL_MULTIPLIER": "1.5",
    }

    policy = IncidentResponsePolicy.from_mapping(mapping)
    state = IncidentResponseState.from_mapping(mapping)

    assert policy.minimum_primary_responders == 2
    assert policy.training_interval_days == 20
    assert state.available_runbooks == ("redis_outage",)
    assert state.training_age_days == 25.0
    assert state.chatops_ready is False
    assert policy.require_major_incident_history is True
    assert policy.major_incident_review_interval_days == 200
    assert policy.major_incident_fail_multiplier == 1.5


def test_publish_incident_response_snapshot() -> None:
    bus = _StubEventBus()
    snapshot = IncidentResponseSnapshot(
        service="emp",
        generated_at=datetime(2025, 3, 1, tzinfo=UTC),
        status=IncidentResponseStatus.warn,
        missing_runbooks=("redis_outage",),
        training_age_days=40.0,
        drill_age_days=20.0,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("incident-a",),
        issues=("Missing runbooks",),
        metadata={"postmortem_backlog_hours": 12.0},
    )

    publish_incident_response_snapshot(bus, snapshot)

    assert bus.events
    event = bus.events[0]
    assert event.type == "telemetry.operational.incident_response"
    assert event.payload["status"] == "warn"


def test_publish_incident_response_snapshot_raises_on_failover_error() -> None:
    snapshot = IncidentResponseSnapshot(
        service="emp",
        generated_at=datetime(2025, 3, 1, tzinfo=UTC),
        status=IncidentResponseStatus.fail,
        missing_runbooks=("redis_outage",),
        training_age_days=None,
        drill_age_days=None,
        primary_oncall=tuple(),
        secondary_oncall=tuple(),
        open_incidents=tuple(),
        issues=("Runtime bus unavailable",),
        metadata={},
    )

    with patch(
        "src.operations.incident_response.publish_event_with_failover",
        side_effect=EventPublishError("runtime", "telemetry.operational.incident_response"),
    ):
        with pytest.raises(EventPublishError):
            publish_incident_response_snapshot(_StubEventBus(), snapshot)


def test_incident_response_major_incident_review_escalation() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("redis_outage",),
        training_interval_days=30,
        drill_interval_days=30,
        minimum_primary_responders=1,
        minimum_secondary_responders=0,
        postmortem_sla_hours=48.0,
        maximum_open_incidents=2,
        require_chatops=False,
        require_major_incident_history=True,
        major_incident_review_interval_days=120,
        major_incident_fail_multiplier=1.5,
    )
    state = IncidentResponseState(
        available_runbooks=("redis_outage",),
        training_age_days=10.0,
        drill_age_days=12.0,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=tuple(),
        postmortem_backlog_hours=2.0,
        chatops_ready=False,
        last_major_incident_age_days=500.0,
    )

    snapshot = evaluate_incident_response(policy, state, service="emp")

    assert snapshot.status is IncidentResponseStatus.fail
    assert any("Major incident review overdue" in issue for issue in snapshot.issues)
    details = snapshot.metadata.get("issue_details")
    assert details is not None
    review_entries = [entry for entry in details if entry["category"] == "major_incident_review"]  # type: ignore[index]
    assert review_entries
    review_detail = review_entries[0]
    assert review_detail["severity"] == "fail"
    assert review_detail["detail"]["fail_threshold_days"] == pytest.approx(180.0)


def test_incident_response_major_incident_review_missing_history_warn() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=tuple(),
        training_interval_days=10,
        drill_interval_days=10,
        minimum_primary_responders=0,
        minimum_secondary_responders=0,
        postmortem_sla_hours=12.0,
        maximum_open_incidents=0,
        require_chatops=False,
        require_major_incident_history=True,
        major_incident_review_interval_days=90,
    )
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=5.0,
        drill_age_days=6.0,
        primary_oncall=tuple(),
        secondary_oncall=tuple(),
        open_incidents=tuple(),
        postmortem_backlog_hours=None,
        chatops_ready=False,
        last_major_incident_age_days=None,
    )

    snapshot = evaluate_incident_response(policy, state, service="emp")

    assert snapshot.status is IncidentResponseStatus.warn
    assert any("No major incident review" in issue for issue in snapshot.issues)
    category_severity = snapshot.metadata.get("issue_category_severity")
    assert category_severity and category_severity["major_incident_review"] == "warn"


def test_incident_response_alert_generation_from_snapshot() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("redis_outage",),
        training_interval_days=14,
        drill_interval_days=14,
        minimum_primary_responders=2,
        minimum_secondary_responders=1,
        postmortem_sla_hours=12.0,
        maximum_open_incidents=0,
        require_chatops=True,
    )
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=40.0,
        drill_age_days=50.0,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("INC-42",),
        postmortem_backlog_hours=30.0,
        chatops_ready=False,
    )

    snapshot = evaluate_incident_response(policy, state, service="emp")
    events = derive_incident_response_alerts(snapshot)

    categories = {event.category: event for event in events}
    assert categories["incident_response.status"].severity is AlertSeverity.critical
    assert categories["incident_response.missing_runbooks"].context["missing_runbooks"] == [
        "redis_outage"
    ]
    assert categories["incident_response.postmortem_backlog"].severity is AlertSeverity.critical
    assert "training overdue" in categories["incident_response.training"].message.lower()
    assert "incident" in categories["incident_response.open_incidents"].message.lower()
    assert categories["incident_response.chatops"].context["chatops_ready"] is False
    issue_event = categories["incident_response.issue"]
    assert "detail" in issue_event.context
    detail = issue_event.context["detail"]
    assert detail["category"] in issue_event.tags


def test_route_incident_response_alerts_dispatches_events() -> None:
    policy = IncidentResponsePolicy(required_runbooks=("redis_outage",))
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=None,
        drill_age_days=None,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("INC-5",),
        postmortem_backlog_hours=0.0,
        chatops_ready=False,
    )

    snapshot = evaluate_incident_response(policy, state, service="emp")
    manager = _StubAlertManager()

    results = route_incident_response_alerts(manager, snapshot)

    assert results, "expected at least one alert dispatch"
    assert all(result.triggered_channels == ("stub",) for result in results)
    assert manager.events


def test_incident_response_metrics_escalation_and_alerts() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=tuple(),
        training_interval_days=30,
        drill_interval_days=30,
        minimum_primary_responders=1,
        minimum_secondary_responders=1,
        postmortem_sla_hours=24.0,
        maximum_open_incidents=1,
        require_chatops=True,
        mtta_warn_minutes=20.0,
        mtta_fail_minutes=40.0,
        mttr_warn_minutes=120.0,
        mttr_fail_minutes=240.0,
        metrics_stale_warn_hours=12.0,
        metrics_stale_fail_hours=24.0,
    )
    state = IncidentResponseState(
        available_runbooks=("redis_outage",),
        training_age_days=10.0,
        drill_age_days=15.0,
        primary_oncall=("alice",),
        secondary_oncall=("bob",),
        open_incidents=("INC-1",),
        postmortem_backlog_hours=8.0,
        chatops_ready=True,
        metrics=IncidentResponseMetrics(
            mtta_minutes=55.0,
            mttr_minutes=250.0,
            acknowledged_incidents=5,
            resolved_incidents=4,
            sample_window_days=7.0,
            data_age_hours=30.0,
        ),
    )

    snapshot = evaluate_incident_response(policy, state)

    assert snapshot.status is IncidentResponseStatus.fail
    assert snapshot.metrics is not None
    assert snapshot.metrics.mtta_minutes == pytest.approx(55.0)
    reliability = snapshot.metadata.get("reliability_metrics")
    assert isinstance(reliability, dict)
    assert reliability["mttr_minutes"] == pytest.approx(250.0)

    events = derive_incident_response_alerts(snapshot)
    categories = {event.category for event in events}
    assert "incident_response.mtta" in categories
    assert "incident_response.mttr" in categories
    assert "incident_response.metrics_staleness" in categories


def test_incident_response_gate_warn_and_block() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("redis_outage",),
        training_interval_days=30,
        drill_interval_days=30,
        minimum_primary_responders=1,
        minimum_secondary_responders=1,
        maximum_open_incidents=1,
        require_chatops=True,
    )
    state = IncidentResponseState(
        available_runbooks=("redis_outage",),
        training_age_days=35.0,
        drill_age_days=10.0,
        primary_oncall=("alice",),
        secondary_oncall=("bob",),
        open_incidents=tuple(),
        postmortem_backlog_hours=2.0,
        chatops_ready=True,
    )

    snapshot = evaluate_incident_response(policy, state)
    gate_default = evaluate_incident_response_gate(snapshot)

    assert isinstance(gate_default, IncidentResponseGateResult)

    assert snapshot.status is IncidentResponseStatus.warn
    assert gate_default.status is IncidentResponseStatus.warn
    assert not gate_default.is_blocking()
    assert gate_default.warnings, "expected training warning to be recorded"

    gate_block = evaluate_incident_response_gate(snapshot, block_on_warn=True)

    assert isinstance(gate_block, IncidentResponseGateResult)

    assert gate_block.status is IncidentResponseStatus.fail
    assert gate_block.is_blocking()
    assert any("training" in reason.lower() for reason in gate_block.blocking_reasons)


def test_incident_response_alerts_include_gate_event() -> None:
    policy = IncidentResponsePolicy(required_runbooks=("redis_outage",))
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=None,
        drill_age_days=None,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("INGEST-1",),
        postmortem_backlog_hours=12.0,
        chatops_ready=False,
    )

    snapshot = evaluate_incident_response(policy, state)
    gate_result = evaluate_incident_response_gate(snapshot)

    events = derive_incident_response_alerts(
        snapshot,
        include_gate_event=True,
        gate_result=gate_result,
    )

    gate_events = [event for event in events if event.category == "incident_response.gate"]
    assert gate_events, "expected gate alert event"
    gate_event = gate_events[0]
    assert gate_event.severity is AlertSeverity.critical
    gate_context = gate_event.context.get("gate")
    assert isinstance(gate_context, dict)
    assert gate_context.get("blocking_reasons")


def test_route_incident_response_alerts_includes_gate_event() -> None:
    policy = IncidentResponsePolicy(required_runbooks=("redis_outage",))
    state = IncidentResponseState(
        available_runbooks=tuple(),
        training_age_days=None,
        drill_age_days=None,
        primary_oncall=("alice",),
        secondary_oncall=tuple(),
        open_incidents=("INGEST-2",),
        postmortem_backlog_hours=18.0,
        chatops_ready=False,
    )

    snapshot = evaluate_incident_response(policy, state)
    manager = _StubAlertManager()

    results = route_incident_response_alerts(
        manager,
        snapshot,
        include_gate_event=True,
    )

    categories = {result.event.category for result in results}
    assert "incident_response.gate" in categories
