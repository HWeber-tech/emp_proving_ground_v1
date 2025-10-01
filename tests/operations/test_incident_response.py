from datetime import UTC, datetime

from src.core.event_bus import Event
from src.operations.incident_response import (
    IncidentResponsePolicy,
    IncidentResponseSnapshot,
    IncidentResponseState,
    IncidentResponseStatus,
    evaluate_incident_response,
    publish_incident_response_snapshot,
)


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._running = True

    def is_running(self) -> bool:  # pragma: no cover - trivial proxy
        return self._running

    def publish_from_sync(self, event: Event) -> None:
        self.events.append(event)
        return True


class _FallbackEventBus(_StubEventBus):
    def publish_from_sync(self, event: Event) -> None:  # pragma: no cover - short path
        self.events.append(event)
        return None


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish_sync(self, event_type: str, payload: object, *, source: str) -> None:
        self.events.append(Event(type=event_type, payload=payload, source=source))


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
    }

    policy = IncidentResponsePolicy.from_mapping(mapping)
    state = IncidentResponseState.from_mapping(mapping)

    assert policy.minimum_primary_responders == 2
    assert policy.training_interval_days == 20
    assert state.available_runbooks == ("redis_outage",)
    assert state.training_age_days == 25.0
    assert state.chatops_ready is False


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


def test_publish_incident_response_snapshot_falls_back_to_global_bus() -> None:
    bus = _FallbackEventBus()
    topic_bus = _StubTopicBus()
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

    publish_incident_response_snapshot(
        bus,
        snapshot,
        global_bus_factory=lambda: topic_bus,
    )

    assert topic_bus.events
    event = topic_bus.events[0]
    assert event.type == "telemetry.operational.incident_response"
    assert event.payload["status"] == "warn"
