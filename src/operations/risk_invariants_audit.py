"""Weekly invariants audit and red-team scenario evaluators.

This module assembles a deterministic weekly audit across risk invariants.  The
audit expects evidence for the core red-team scenarios (extreme volatility,
symbol halts, and bad market-data prints) and emits a structured snapshot that
captures runtime coverage, guardrail posture, and recency checks.  The helpers
lean on the existing simulation invariant assessment utilities so operators can
wire the audit into scheduled jobs or notebooks without re-implementing
guardrail logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Mapping, MutableMapping, Sequence

from src.runtime.paper_simulation import PaperTradingSimulationReport
from src.runtime.simulation_invariants import (
    SimulationInvariantAssessment,
    assess_simulation_invariants,
)

__all__ = [
    "AuditStatus",
    "InvariantScenarioEvidence",
    "InvariantScenarioSnapshot",
    "WeeklyInvariantsAudit",
    "REQUIRED_SCENARIOS",
    "audit_weekly_invariants",
]


DEFAULT_REQUIRED_RUNTIME_SECONDS = float(60 * 60)  # one hour
DEFAULT_RECENCY_THRESHOLD = timedelta(days=7)


REQUIRED_SCENARIOS: Mapping[str, str] = {
    "extreme_volatility": "Shock the order book with rapid price swings",
    "symbol_halt": "Primary listing halt and reopen flow",
    "bad_prints": "Erroneous market-data prints propagating through the loop",
}


class AuditStatus(StrEnum):
    """Severity ladder for invariant audit components."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[AuditStatus, int] = {
    AuditStatus.ok: 0,
    AuditStatus.warn: 1,
    AuditStatus.fail: 2,
}


def _combine_status(current: AuditStatus, candidate: AuditStatus) -> AuditStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


def _normalise_timestamp(candidate: datetime) -> datetime:
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=UTC)
    return candidate.astimezone(UTC)


@dataclass(frozen=True)
class InvariantScenarioEvidence:
    """Input evidence describing a specific red-team scenario run."""

    scenario: str
    report: PaperTradingSimulationReport
    executed_at: datetime
    required_runtime_seconds: float = DEFAULT_REQUIRED_RUNTIME_SECONDS
    simulated_tick_seconds: float = 0.5
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario:
            raise ValueError("scenario must be provided")
        if self.required_runtime_seconds <= 0:
            raise ValueError("required_runtime_seconds must be positive")
        if self.simulated_tick_seconds <= 0:
            raise ValueError("simulated_tick_seconds must be positive")


@dataclass(frozen=True)
class InvariantScenarioSnapshot:
    """Structured snapshot summarising a scenario audit outcome."""

    scenario: str
    status: AuditStatus
    summary: str
    executed_at: datetime | None
    assessment: SimulationInvariantAssessment | None
    messages: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "scenario": self.scenario,
            "status": self.status.value,
            "summary": self.summary,
            "messages": list(self.messages),
            "metadata": dict(self.metadata),
        }
        if self.executed_at is not None:
            payload["executed_at"] = self.executed_at.astimezone(UTC).isoformat()
        if self.assessment is not None:
            payload["assessment"] = {
                "simulated_runtime_seconds": self.assessment.simulated_runtime_seconds,
                "tick_count": self.assessment.tick_count,
                "guardrail_violations": self.assessment.guardrail_violations,
                "guardrail_near_misses": self.assessment.guardrail_near_misses,
                "last_guardrail_severity": self.assessment.last_guardrail_severity,
            }
        return payload

    def to_markdown_row(self) -> str:
        executed = self.executed_at.isoformat() if self.executed_at else "(missing)"
        return f"| {self.scenario} | {self.status.value.upper()} | {self.summary} | {executed} |"


@dataclass(frozen=True)
class WeeklyInvariantsAudit:
    """Aggregate view of the weekly invariant audit."""

    status: AuditStatus
    generated_at: datetime
    scenarios: tuple[InvariantScenarioSnapshot, ...]
    missing_scenarios: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "missing_scenarios": list(self.missing_scenarios),
        }

    def to_markdown(self) -> str:
        rows = [
            "| Scenario | Status | Summary | Executed At |",
            "| --- | --- | --- | --- |",
        ]
        for scenario in self.scenarios:
            rows.append(scenario.to_markdown_row())
        return "\n".join(rows)


def audit_weekly_invariants(
    evidence: Sequence[InvariantScenarioEvidence],
    *,
    required_scenarios: Mapping[str, str] = REQUIRED_SCENARIOS,
    recency_threshold: timedelta = DEFAULT_RECENCY_THRESHOLD,
    now: datetime | None = None,
) -> WeeklyInvariantsAudit:
    """Evaluate weekly invariant posture across red-team scenarios."""

    audit_time = _normalise_timestamp(now or datetime.now(tz=UTC))

    latest: dict[str, InvariantScenarioEvidence] = {}
    for entry in evidence:
        candidate_time = _normalise_timestamp(entry.executed_at)
        prior = latest.get(entry.scenario)
        if prior is None or candidate_time > _normalise_timestamp(prior.executed_at):
            latest[entry.scenario] = entry

    snapshots: list[InvariantScenarioSnapshot] = []
    missing: list[str] = []
    status = AuditStatus.ok

    for scenario, description in required_scenarios.items():
        entry = latest.get(scenario)
        if entry is None:
            summary = f"scenario not executed – {description}"
            snapshot = InvariantScenarioSnapshot(
                scenario=scenario,
                status=AuditStatus.fail,
                summary=summary,
                executed_at=None,
                assessment=None,
                messages=(summary,),
                metadata={"description": description},
            )
            missing.append(scenario)
        else:
            snapshot = _evaluate_scenario(
                entry,
                description=description,
                now=audit_time,
                recency_threshold=recency_threshold,
            )
        snapshots.append(snapshot)
        status = _combine_status(status, snapshot.status)

    extra_scenarios = [
        scenario
        for scenario in latest
        if scenario not in required_scenarios
    ]
    for scenario in sorted(extra_scenarios):
        snapshot = _evaluate_scenario(
            latest[scenario],
            description=None,
            now=audit_time,
            recency_threshold=recency_threshold,
        )
        snapshots.append(snapshot)
        status = _combine_status(status, snapshot.status)

    return WeeklyInvariantsAudit(
        status=status,
        generated_at=audit_time,
        scenarios=tuple(snapshots),
        missing_scenarios=tuple(missing),
    )


def _evaluate_scenario(
    entry: InvariantScenarioEvidence,
    *,
    description: str | None,
    now: datetime,
    recency_threshold: timedelta,
) -> InvariantScenarioSnapshot:
    executed_at = _normalise_timestamp(entry.executed_at)
    assessment = assess_simulation_invariants(
        entry.report,
        simulated_tick_seconds=entry.simulated_tick_seconds,
    )

    metadata: dict[str, object] = dict(entry.metadata)
    metadata.setdefault("required_runtime_seconds", entry.required_runtime_seconds)
    metadata.setdefault("simulated_tick_seconds", entry.simulated_tick_seconds)
    metadata.update(
        {
            "simulated_runtime_seconds": assessment.simulated_runtime_seconds,
            "tick_count": assessment.tick_count,
            "guardrail_violations": assessment.guardrail_violations,
            "guardrail_near_misses": assessment.guardrail_near_misses,
        }
    )
    if description:
        metadata.setdefault("description", description)

    status = AuditStatus.ok
    messages: list[str] = []

    runtime_shortfall = max(
        0.0,
        float(entry.required_runtime_seconds) - float(assessment.simulated_runtime_seconds),
    )
    if runtime_shortfall > 0.0:
        metadata["runtime_shortfall_seconds"] = runtime_shortfall
        messages.append(
            f"runtime shortfall {runtime_shortfall:.1f}s (required {entry.required_runtime_seconds:.1f}s)"
        )
        status = _combine_status(status, AuditStatus.fail)

    if assessment.guardrail_violations > 0:
        messages.append(
            f"guardrail violations observed ({assessment.guardrail_violations})"
        )
        status = _combine_status(status, AuditStatus.fail)

    if assessment.last_guardrail_severity and assessment.last_guardrail_severity.strip().lower() == "violation":
        messages.append("last guardrail incident flagged a violation")
        status = _combine_status(status, AuditStatus.fail)

    if assessment.guardrail_near_misses > 0:
        messages.append(
            f"guardrail near misses observed ({assessment.guardrail_near_misses})"
        )
        status = _combine_status(status, AuditStatus.warn)

    age = now - executed_at
    metadata["age_seconds"] = age.total_seconds()
    if age > recency_threshold:
        stale_seconds = age.total_seconds() - recency_threshold.total_seconds()
        metadata["stale_by_seconds"] = stale_seconds
        messages.append(
            f"scenario execution stale by {stale_seconds:.0f}s"
        )
        status = _combine_status(status, AuditStatus.warn)

    if assessment.simulated_runtime_seconds <= 0:
        messages.append("simulated runtime zero; check execution stats")
        status = _combine_status(status, AuditStatus.warn)

    if not messages:
        messages.append("scenario completed without guardrail incidents")

    summary = messages[0]

    return InvariantScenarioSnapshot(
        scenario=entry.scenario,
        status=status,
        summary=summary,
        executed_at=executed_at,
        assessment=assessment,
        messages=tuple(messages),
        metadata=metadata,
    )
