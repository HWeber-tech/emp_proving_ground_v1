"""Chaos engineering utilities for incident response simulations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from random import Random
from typing import Callable, Mapping, Sequence

from src.operations.incident_response import (
    IncidentResponseMetrics,
    IncidentResponsePolicy,
    IncidentResponseSnapshot,
    IncidentResponseState,
    IncidentResponseStatus,
    evaluate_incident_response,
)


logger = logging.getLogger(__name__)


class ChaosEventType(StrEnum):
    """Supported incident chaos scenarios."""

    primary_responder_outage = "primary_responder_outage"
    secondary_responder_overload = "secondary_responder_overload"
    runbook_corruption = "runbook_corruption"
    chatops_failure = "chatops_failure"
    metrics_stale = "metrics_stale"
    incident_surge = "incident_surge"
    postmortem_backlog = "postmortem_backlog"
    training_lapse = "training_lapse"


_DEFAULT_EVENT_SEVERITY: Mapping[ChaosEventType, IncidentResponseStatus] = {
    ChaosEventType.primary_responder_outage: IncidentResponseStatus.fail,
    ChaosEventType.secondary_responder_overload: IncidentResponseStatus.warn,
    ChaosEventType.runbook_corruption: IncidentResponseStatus.fail,
    ChaosEventType.chatops_failure: IncidentResponseStatus.warn,
    ChaosEventType.metrics_stale: IncidentResponseStatus.warn,
    ChaosEventType.incident_surge: IncidentResponseStatus.fail,
    ChaosEventType.postmortem_backlog: IncidentResponseStatus.warn,
    ChaosEventType.training_lapse: IncidentResponseStatus.warn,
}


_STATUS_ORDER: Mapping[IncidentResponseStatus, int] = {
    IncidentResponseStatus.ok: 0,
    IncidentResponseStatus.warn: 1,
    IncidentResponseStatus.fail: 2,
}


def _escalate(
    current: IncidentResponseStatus, candidate: IncidentResponseStatus
) -> IncidentResponseStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_names(value: object | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        items = [part.strip() for part in value.replace(";", ",").split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return tuple(result)
    return (str(value),)


def _resolve_members(
    existing: tuple[str, ...], candidates: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not existing:
        return existing, tuple()
    if candidates:
        lookup = {value.lower(): value for value in existing}
        selected = [
            lookup[name.lower()]
            for name in candidates
            if name.lower() in lookup
        ]
    else:
        selected = list(existing)
    remaining = tuple(member for member in existing if member not in selected)
    return remaining, tuple(selected)


def _ensure_metrics(state: IncidentResponseState) -> IncidentResponseMetrics:
    metrics = state.metrics
    if metrics is None:
        return IncidentResponseMetrics()
    return metrics


@dataclass(frozen=True)
class ChaosEvent:
    """Declarative chaos scenario applied to the incident state."""

    event_type: ChaosEventType
    severity: IncidentResponseStatus | None = None
    description: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def resolved_severity(self) -> IncidentResponseStatus:
        """Return the effective severity for the chaos event."""

        severity = self.severity
        if severity is not None:
            return severity
        return _DEFAULT_EVENT_SEVERITY[self.event_type]

    def resolved_description(self) -> str:
        """Return a human readable description for the chaos event."""

        if self.description:
            return self.description
        return self.event_type.replace("_", " ")

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event_type": self.event_type.value,
            "severity": self.resolved_severity().value,
            "description": self.resolved_description(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SimulatedChaosEvent:
    """Event outcome recorded during the simulation."""

    sequence: int
    applied_at: datetime
    event_type: ChaosEventType
    severity: IncidentResponseStatus
    description: str
    impact: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "applied_at": self.applied_at.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "impact": dict(self.impact),
        }


@dataclass(frozen=True)
class IncidentSimulationResult:
    """Snapshot describing the outcome of a chaos simulation."""

    scenario: str
    generated_at: datetime
    status: IncidentResponseStatus
    snapshot: IncidentResponseSnapshot
    events: tuple[SimulatedChaosEvent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def worst_event_severity(self) -> IncidentResponseStatus:
        severity = IncidentResponseStatus.ok
        for event in self.events:
            severity = _escalate(severity, event.severity)
        return severity

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "scenario": self.scenario,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "snapshot": self.snapshot.as_dict(),
            "events": [event.as_dict() for event in self.events],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        lines = [
            f"### Incident simulation – {self.scenario}",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Simulation status: {self.status.value.upper()}",
            f"- Worst event severity: {self.worst_event_severity.value.upper()}",
            "",
        ]
        if self.events:
            lines.append("**Chaos events:**")
            for event in self.events:
                lines.append(
                    f"- ({event.sequence}) {event.event_type.value}: "
                    f"{event.severity.value.upper()} – {event.description}"
                )
        else:
            lines.append("No chaos events applied.")
        lines.append("")
        lines.append("**Incident response snapshot:**")
        lines.append(self.snapshot.to_markdown())
        return "\n".join(lines)


@dataclass(frozen=True)
class ChaosScenario:
    """Declarative chaos drill describing the events to exercise."""

    name: str
    events: tuple[ChaosEvent, ...] = tuple()
    description: str | None = None
    objectives: tuple[str, ...] = tuple()
    metadata: Mapping[str, object] = field(default_factory=dict)
    randomise_events: bool = False

    def resolve_events(self, rng: Random | None = None) -> tuple[ChaosEvent, ...]:
        """Return the ordered chaos events for this scenario."""

        if not self.events:
            return tuple()
        if self.randomise_events and rng is not None:
            events = list(self.events)
            rng.shuffle(events)
            return tuple(events)
        return self.events

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "events": [event.as_dict() for event in self.events],
        }
        if self.description:
            payload["description"] = self.description
        if self.objectives:
            payload["objectives"] = list(self.objectives)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.randomise_events:
            payload["randomise_events"] = True
        return payload


@dataclass(frozen=True)
class ChaosScenarioOutcome:
    """Outcome captured after running a chaos scenario."""

    scenario: ChaosScenario
    result: IncidentSimulationResult

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario.as_dict(),
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True)
class ChaosCampaignResult:
    """Aggregated outcome for a collection of chaos scenarios."""

    executed_at: datetime
    status: IncidentResponseStatus
    outcomes: tuple[ChaosScenarioOutcome, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def worst_scenario_severity(self) -> IncidentResponseStatus:
        severity = IncidentResponseStatus.ok
        for outcome in self.outcomes:
            severity = _escalate(severity, outcome.result.status)
        return severity

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "executed_at": self.executed_at.isoformat(),
            "status": self.status.value,
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        lines = [
            "## Chaos campaign summary",
            f"- Executed: {self.executed_at.isoformat()}",
            f"- Overall status: {self.status.value.upper()}",
            f"- Scenarios executed: {len(self.outcomes)}",
            "",
        ]
        if self.outcomes:
            lines.append("### Scenario outcomes")
            for outcome in self.outcomes:
                result = outcome.result
                lines.append(
                    f"- {outcome.scenario.name}: "
                    f"{result.status.value.upper()} (worst event {result.worst_event_severity.value.upper()})"
                )
        else:
            lines.append("No scenarios executed.")
        return "\n".join(lines)


StateFactory = Callable[[ChaosScenario], IncidentResponseState]


def run_chaos_campaign(
    policy: IncidentResponsePolicy,
    base_state: IncidentResponseState,
    scenarios: Sequence[ChaosScenario],
    *,
    state_factory: StateFactory | None = None,
    random_order: bool = False,
    random_seed: int | None = None,
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ChaosCampaignResult:
    """Execute a series of chaos scenarios and aggregate the outcomes."""

    moment = now or datetime.now(tz=UTC)
    rng = Random(random_seed) if (random_order or any(s.randomise_events for s in scenarios)) else None
    scenario_list = list(scenarios)

    if random_order and rng is not None:
        rng.shuffle(scenario_list)

    outcomes: list[ChaosScenarioOutcome] = []
    status = IncidentResponseStatus.ok

    for index, scenario in enumerate(scenario_list):
        scenario_rng = rng if scenario.randomise_events else None
        scenario_events = scenario.resolve_events(scenario_rng)
        scenario_state = state_factory(scenario) if state_factory else base_state
        scenario_metadata: dict[str, object] = {
            "scenario": scenario.name,
            "sequence": index + 1,
        }
        if metadata:
            scenario_metadata.update(metadata)
        if scenario.metadata:
            scenario_metadata.update(scenario.metadata)
        if scenario.objectives:
            scenario_metadata.setdefault("objectives", list(scenario.objectives))

        result = simulate_incident_response(
            policy,
            scenario_state,
            events=scenario_events,
            scenario=scenario.name,
            now=moment + timedelta(minutes=index),
            metadata=scenario_metadata,
        )
        status = _escalate(status, result.status)
        outcomes.append(ChaosScenarioOutcome(scenario=scenario, result=result))

    campaign_metadata: dict[str, object] = {
        "scenario_names": [outcome.scenario.name for outcome in outcomes],
        "random_order": random_order,
    }
    if random_seed is not None:
        campaign_metadata["random_seed"] = random_seed
    if metadata:
        campaign_metadata.update(metadata)

    return ChaosCampaignResult(
        executed_at=moment,
        status=status,
        outcomes=tuple(outcomes),
        metadata=campaign_metadata,
    )


def simulate_incident_response(
    policy: IncidentResponsePolicy,
    state: IncidentResponseState,
    *,
    events: Sequence[ChaosEvent] = (),
    scenario: str = "chaos_drill",
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IncidentSimulationResult:
    """Execute a chaos drill against the supplied incident state."""

    moment = now or datetime.now(tz=UTC)
    applied_events: list[SimulatedChaosEvent] = []
    current_state = state
    total_events = len(events)

    for index, event in enumerate(events, start=1):
        applied_at = moment - timedelta(minutes=max(total_events - index, 0) * 5)
        current_state, impact = _apply_event(policy, current_state, event)
        applied_events.append(
            SimulatedChaosEvent(
                sequence=index,
                applied_at=applied_at,
                event_type=event.event_type,
                severity=event.resolved_severity(),
                description=event.resolved_description(),
                impact=impact,
            )
        )

    chaos_metadata = [event.as_dict() for event in applied_events]
    evaluation_metadata: dict[str, object] = {
        "scenario": scenario,
        "chaos_events": chaos_metadata,
    }
    if metadata:
        evaluation_metadata.update(metadata)

    snapshot = evaluate_incident_response(
        policy,
        current_state,
        service=scenario,
        now=moment,
        metadata=evaluation_metadata,
    )

    worst_event = IncidentResponseStatus.ok
    for event in applied_events:
        worst_event = _escalate(worst_event, event.severity)

    result_metadata: dict[str, object] = {
        "scenario": scenario,
        "event_types": [event.event_type.value for event in applied_events],
        "worst_event_severity": worst_event.value,
    }
    if metadata:
        result_metadata.update(metadata)

    status = snapshot.status
    return IncidentSimulationResult(
        scenario=scenario,
        generated_at=moment,
        status=status,
        snapshot=snapshot,
        events=tuple(applied_events),
        metadata=result_metadata,
    )


def _apply_event(
    policy: IncidentResponsePolicy,
    state: IncidentResponseState,
    event: ChaosEvent,
) -> tuple[IncidentResponseState, Mapping[str, object]]:
    impact: dict[str, object] = {}
    event_metadata = event.metadata or {}

    if event.event_type is ChaosEventType.primary_responder_outage:
        candidates = _normalise_names(event_metadata.get("responders"))
        remaining, removed = _resolve_members(state.primary_oncall, candidates)
        impact["removed_primary"] = list(removed)
        impact["remaining_primary"] = list(remaining)
        state = replace(state, primary_oncall=remaining)

    elif event.event_type is ChaosEventType.secondary_responder_overload:
        candidates = _normalise_names(event_metadata.get("responders"))
        remaining, removed = _resolve_members(state.secondary_oncall, candidates)
        impact["removed_secondary"] = list(removed)
        impact["remaining_secondary"] = list(remaining)
        state = replace(state, secondary_oncall=remaining)

    elif event.event_type is ChaosEventType.runbook_corruption:
        candidates = _normalise_names(event_metadata.get("runbooks"))
        available_lookup = {name.lower(): name for name in state.available_runbooks}
        removed: list[str] = []
        if candidates:
            for candidate in candidates:
                lowered = candidate.lower()
                if lowered in available_lookup:
                    removed.append(available_lookup[lowered])
        else:
            for required in policy.required_runbooks:
                lowered = required.lower()
                if lowered in available_lookup:
                    removed.append(available_lookup[lowered])
            if not removed and state.available_runbooks:
                removed.append(state.available_runbooks[0])
        remaining = tuple(
            runbook for runbook in state.available_runbooks if runbook not in removed
        )
        impact["removed_runbooks"] = list(removed)
        impact["remaining_runbooks"] = list(remaining)
        state = replace(state, available_runbooks=remaining)

    elif event.event_type is ChaosEventType.chatops_failure:
        impact["chatops_ready"] = False
        state = replace(state, chatops_ready=False)

    elif event.event_type is ChaosEventType.metrics_stale:
        metrics = _ensure_metrics(state)
        age_hours = event_metadata.get("age_hours")
        if age_hours is None:
            age_hours = policy.metrics_stale_fail_hours or policy.metrics_stale_warn_hours or 48.0
        try:
            age_value = float(age_hours)
        except (TypeError, ValueError):
            age_value = float(policy.metrics_stale_fail_hours or 48.0)
        mtta_minutes = event_metadata.get("mtta_minutes")
        mttr_minutes = event_metadata.get("mttr_minutes")
        metrics = replace(
            metrics,
            data_age_hours=float(age_value),
            mtta_minutes=(
                float(mtta_minutes)
                if mtta_minutes is not None and _is_number(mtta_minutes)
                else metrics.mtta_minutes
            ),
            mttr_minutes=(
                float(mttr_minutes)
                if mttr_minutes is not None and _is_number(mttr_minutes)
                else metrics.mttr_minutes
            ),
        )
        impact["data_age_hours"] = metrics.data_age_hours
        if metrics.mtta_minutes is not None:
            impact["mtta_minutes"] = metrics.mtta_minutes
        if metrics.mttr_minutes is not None:
            impact["mttr_minutes"] = metrics.mttr_minutes
        state = replace(state, metrics=metrics)

    elif event.event_type is ChaosEventType.incident_surge:
        base_prefix = str(event_metadata.get("prefix") or event.event_type.value)
        try:
            count = int(event_metadata.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        existing = list(state.open_incidents)
        added: list[str] = []
        start_index = len(existing) + 1
        for offset in range(count):
            incident_id = f"{base_prefix}-{start_index + offset:03d}"
            existing.append(incident_id)
            added.append(incident_id)
        impact["added_incidents"] = added
        state = replace(state, open_incidents=tuple(existing))

    elif event.event_type is ChaosEventType.postmortem_backlog:
        hours = event_metadata.get("hours")
        if hours is None:
            hours = max((policy.postmortem_sla_hours or 24.0) * 4, 24.0)
        try:
            backlog = float(hours)
        except (TypeError, ValueError):
            backlog = float(policy.postmortem_sla_hours or 24.0) * 4
        impact["postmortem_backlog_hours"] = backlog
        state = replace(state, postmortem_backlog_hours=backlog)

    elif event.event_type is ChaosEventType.training_lapse:
        training_age = event_metadata.get("training_age_days")
        drill_age = event_metadata.get("drill_age_days")
        if training_age is None:
            training_age = (policy.training_interval_days or 90) * 2
        if drill_age is None:
            drill_age = (policy.drill_interval_days or 60) * 2
        try:
            training_value = float(training_age)
        except (TypeError, ValueError):
            training_value = float(policy.training_interval_days or 90) * 2
        try:
            drill_value = float(drill_age)
        except (TypeError, ValueError):
            drill_value = float(policy.drill_interval_days or 60) * 2
        impact["training_age_days"] = training_value
        impact["drill_age_days"] = drill_value
        state = replace(
            state,
            training_age_days=training_value,
            drill_age_days=drill_value,
        )

    else:  # pragma: no cover - defensive branch for new event types
        logger.debug("Unhandled chaos event type", extra={"event_type": event.event_type})

    return state, impact


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True

