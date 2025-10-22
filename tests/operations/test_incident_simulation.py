from datetime import UTC, datetime

from src.operations.incident_response import (
    IncidentResponseMetrics,
    IncidentResponsePolicy,
    IncidentResponseState,
    IncidentResponseStatus,
)
from src.operations.incident_simulation import (
    ChaosCampaignResult,
    ChaosEvent,
    ChaosEventType,
    ChaosScenario,
    run_chaos_campaign,
    simulate_incident_response,
)


def test_simulation_escalates_and_records_impacts() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("core", "disaster"),
        training_interval_days=90,
        drill_interval_days=60,
        postmortem_sla_hours=24.0,
    )
    state = IncidentResponseState(
        available_runbooks=("core", "disaster"),
        training_age_days=10.0,
        drill_age_days=5.0,
        primary_oncall=("alice", "bob"),
        secondary_oncall=("charlie", "dora"),
        chatops_ready=True,
        metrics=IncidentResponseMetrics(
            mtta_minutes=12.0,
            mttr_minutes=90.0,
            acknowledged_incidents=5,
            resolved_incidents=5,
            sample_window_days=30.0,
            data_age_hours=2.0,
        ),
    )

    events = (
        ChaosEvent(
            ChaosEventType.primary_responder_outage,
            metadata={"responders": ["alice"]},
        ),
        ChaosEvent(
            ChaosEventType.runbook_corruption,
            metadata={"runbooks": ["disaster"]},
        ),
        ChaosEvent(
            ChaosEventType.incident_surge,
            metadata={"count": 2, "prefix": "chaos"},
        ),
    )

    result = simulate_incident_response(
        policy,
        state,
        events=events,
        scenario="test_drill",
    )

    assert result.status is IncidentResponseStatus.fail
    assert "disaster" in result.snapshot.missing_runbooks
    assert len(result.snapshot.open_incidents) == 2
    assert result.metadata["worst_event_severity"] == "fail"

    first_event = result.events[0]
    assert first_event.event_type is ChaosEventType.primary_responder_outage
    assert first_event.impact["removed_primary"] == ["alice"]
    assert "alice" not in result.snapshot.primary_oncall

    runbook_event = result.events[1]
    assert runbook_event.impact["removed_runbooks"] == ["disaster"]

    surge_event = result.events[2]
    assert surge_event.impact["added_incidents"]
    assert len(surge_event.impact["added_incidents"]) == 2


def test_metrics_stale_event_escalates_failure() -> None:
    policy = IncidentResponsePolicy(
        metrics_stale_warn_hours=12.0,
        metrics_stale_fail_hours=24.0,
    )
    state = IncidentResponseState(
        primary_oncall=("alice",),
        secondary_oncall=("bob",),
        chatops_ready=True,
        metrics=IncidentResponseMetrics(
            mtta_minutes=15.0,
            mttr_minutes=120.0,
            acknowledged_incidents=3,
            resolved_incidents=3,
            sample_window_days=30.0,
            data_age_hours=2.0,
        ),
    )

    event = ChaosEvent(
        ChaosEventType.metrics_stale,
        metadata={"age_hours": 36, "mtta_minutes": 90, "mttr_minutes": 480},
    )

    result = simulate_incident_response(
        policy,
        state,
        events=[event],
        now=datetime.now(tz=UTC),
    )

    assert result.status is IncidentResponseStatus.fail
    assert any("metrics age 36.0 hours" in issue for issue in result.snapshot.issues)
    assert result.events[0].impact["data_age_hours"] == 36.0

    payload = result.as_dict()
    assert payload["status"] == "fail"
    assert payload["events"][0]["event_type"] == "metrics_stale"


def test_chaos_campaign_aggregates_scenarios_and_metadata() -> None:
    policy = IncidentResponsePolicy(
        required_runbooks=("core",),
        minimum_primary_responders=1,
        minimum_secondary_responders=1,
        metrics_stale_fail_hours=24.0,
    )
    base_state = IncidentResponseState(
        available_runbooks=("core",),
        primary_oncall=("alice", "bob"),
        secondary_oncall=("charlie",),
        metrics=IncidentResponseMetrics(
            mtta_minutes=10.0,
            mttr_minutes=80.0,
            acknowledged_incidents=4,
            resolved_incidents=4,
            sample_window_days=14.0,
            data_age_hours=4.0,
        ),
    )

    runbook_failure = ChaosScenario(
        name="responder_gap",
        events=(
            ChaosEvent(
                ChaosEventType.primary_responder_outage,
                metadata={"responders": ["alice"]},
            ),
        ),
        objectives=("Verify secondary responder coverage",),
        metadata={"playbook": "pagerduty"},
    )
    metrics_stale = ChaosScenario(
        name="metrics_stale",
        events=(
            ChaosEvent(
                ChaosEventType.metrics_stale,
                metadata={"age_hours": 72},
            ),
        ),
        description="Validate stale telemetry escalations",
    )

    campaign = run_chaos_campaign(
        policy,
        base_state,
        [runbook_failure, metrics_stale],
        metadata={"environment": "staging"},
    )

    assert isinstance(campaign, ChaosCampaignResult)
    assert len(campaign.outcomes) == 2
    assert campaign.status is IncidentResponseStatus.fail
    assert campaign.metadata["scenario_names"] == ["responder_gap", "metrics_stale"]
    assert campaign.metadata["environment"] == "staging"

    first_outcome = campaign.outcomes[0]
    assert first_outcome.result.metadata["objectives"] == [
        "Verify secondary responder coverage"
    ]
    assert first_outcome.result.events[0].impact["removed_primary"] == ["alice"]
    assert campaign.outcomes[1].scenario.description == "Validate stale telemetry escalations"


def test_chaos_campaign_randomises_events_and_uses_state_factory() -> None:
    policy = IncidentResponsePolicy(
        minimum_primary_responders=1,
        minimum_secondary_responders=1,
    )
    base_state = IncidentResponseState(
        primary_oncall=("alice",),
        secondary_oncall=("bob",),
    )

    scenario = ChaosScenario(
        name="rotation",
        events=(
            ChaosEvent(
                ChaosEventType.primary_responder_outage,
                metadata={"responders": ["alice"]},
            ),
            ChaosEvent(
                ChaosEventType.secondary_responder_overload,
                metadata={"responders": ["charlie"]},
            ),
        ),
        randomise_events=True,
    )

    def build_state(_scenario: ChaosScenario) -> IncidentResponseState:
        return IncidentResponseState(
            primary_oncall=("alice", "eve"),
            secondary_oncall=("charlie", "bob"),
        )

    campaign = run_chaos_campaign(
        policy,
        base_state,
        [scenario],
        state_factory=build_state,
        random_seed=3,
    )

    outcome = campaign.outcomes[0]
    assert outcome.result.events[0].event_type is ChaosEventType.secondary_responder_overload
    snapshot = outcome.result.snapshot
    assert snapshot.primary_oncall == ("eve",)
    assert "charlie" not in snapshot.secondary_oncall
    assert campaign.metadata["random_seed"] == 3
