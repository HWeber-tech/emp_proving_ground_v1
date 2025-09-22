"""Failover drill helpers aligned with the data-backbone roadmap."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Awaitable, Callable, Mapping, Sequence

from src.data_foundation.ingest.failover import (
    IngestFailoverDecision,
    IngestFailoverPolicy,
    decide_ingest_failover,
)
from src.data_foundation.ingest.health import (
    IngestHealthReport,
    IngestHealthStatus,
    evaluate_ingest_health,
)
from src.data_foundation.ingest.timescale_pipeline import TimescaleBackbonePlan
from src.data_foundation.persist.timescale import TimescaleIngestResult


class FailoverDrillStatus(StrEnum):
    """Severity levels for failover drill components."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[FailoverDrillStatus, int] = {
    FailoverDrillStatus.ok: 0,
    FailoverDrillStatus.warn: 1,
    FailoverDrillStatus.fail: 2,
}


_HEALTH_TO_DRILL: dict[IngestHealthStatus, FailoverDrillStatus] = {
    IngestHealthStatus.ok: FailoverDrillStatus.ok,
    IngestHealthStatus.warn: FailoverDrillStatus.warn,
    IngestHealthStatus.error: FailoverDrillStatus.fail,
}


def _combine_status(
    current: FailoverDrillStatus, candidate: FailoverDrillStatus
) -> FailoverDrillStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class FailoverDrillComponent:
    """Individual component recorded in a failover drill."""

    name: str
    status: FailoverDrillStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class FailoverDrillSnapshot:
    """Aggregated snapshot describing a failover drill outcome."""

    status: FailoverDrillStatus
    generated_at: datetime
    scenario: str
    components: tuple[FailoverDrillComponent, ...]
    health_report: IngestHealthReport
    failover_decision: IngestFailoverDecision
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "scenario": self.scenario,
            "components": [component.as_dict() for component in self.components],
            "health_report": self.health_report.as_dict(),
            "failover_decision": self.failover_decision.as_dict(),
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        if not self.components:
            return "| Component | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            rows.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        return "\n".join(rows)


async def execute_failover_drill(
    *,
    plan: TimescaleBackbonePlan,
    results: Mapping[str, TimescaleIngestResult],
    fail_dimensions: Sequence[str],
    scenario: str = "timescale_failover",
    failover_policy: IngestFailoverPolicy | None = None,
    fallback: Callable[[], Awaitable[None] | None] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FailoverDrillSnapshot:
    """Simulate a Timescale failure and confirm failover behaviour."""

    generated_at = datetime.now(tz=UTC)
    simulated: dict[str, TimescaleIngestResult] = dict(results)
    targeted: list[str] = []

    for dimension in fail_dimensions:
        targeted.append(dimension)
        prior = results.get(dimension)
        simulated[dimension] = TimescaleIngestResult.empty(
            dimension=dimension,
            source=prior.source if prior else None,
        )

    health_metadata = dict(metadata) if metadata else {}
    health_metadata.update(
        {
            "drill": {
                "scenario": scenario,
                "dimensions": list(targeted),
            }
        }
    )

    health_report = evaluate_ingest_health(
        simulated,
        plan=plan,
        metadata=health_metadata,
    )
    decision = decide_ingest_failover(
        health_report,
        plan=plan,
        policy=failover_policy,
    )

    fallback_error: str | None = None
    fallback_executed = False
    if decision.should_failover and fallback is not None:
        fallback_executed = True
        try:
            outcome = fallback()
            if inspect.isawaitable(outcome):
                await outcome
        except Exception as exc:  # pragma: no cover - defensive logging
            fallback_error = str(exc)

    components: list[FailoverDrillComponent] = []

    health_component = FailoverDrillComponent(
        name="health",
        status=_HEALTH_TO_DRILL[health_report.status],
        summary=f"health status {health_report.status.value}",
        metadata={
            "checks": [check.as_dict() for check in health_report.checks],
        },
    )
    components.append(health_component)

    if decision.should_failover:
        failover_status = (
            FailoverDrillStatus.ok if decision.triggered_dimensions else FailoverDrillStatus.warn
        )
        failover_summary = decision.reason or "failover triggered"
    else:
        failover_status = FailoverDrillStatus.warn if targeted else FailoverDrillStatus.ok
        failover_summary = "failover not triggered"

    failover_component = FailoverDrillComponent(
        name="failover",
        status=failover_status,
        summary=failover_summary,
        metadata={
            "targeted": list(targeted),
            "triggered": list(decision.triggered_dimensions),
            "optional_triggers": list(decision.optional_triggers),
        },
    )
    components.append(failover_component)

    if decision.should_failover:
        if fallback is None:
            fallback_status = FailoverDrillStatus.warn
            fallback_summary = "fallback not provided"
        elif fallback_error is not None:
            fallback_status = FailoverDrillStatus.fail
            fallback_summary = f"fallback failed: {fallback_error}"
        else:
            fallback_status = FailoverDrillStatus.ok
            fallback_summary = "fallback executed"
    else:
        if targeted:
            fallback_status = FailoverDrillStatus.warn
            fallback_summary = "no fallback required for simulated outage"
        else:
            fallback_status = FailoverDrillStatus.ok
            fallback_summary = "no failover scenario configured"

    fallback_component = FailoverDrillComponent(
        name="fallback",
        status=fallback_status,
        summary=fallback_summary,
        metadata={
            "executed": fallback_executed,
            "error": fallback_error,
        },
    )
    components.append(fallback_component)

    overall = FailoverDrillStatus.ok
    for status in (failover_component.status, fallback_component.status):
        overall = _combine_status(overall, status)

    snapshot_metadata = dict(metadata) if metadata else {}
    snapshot_metadata.setdefault("targeted_dimensions", list(targeted))
    snapshot_metadata.setdefault("failover_triggered", decision.should_failover)
    snapshot_metadata.setdefault(
        "fallback", {"executed": fallback_executed, "error": fallback_error}
    )

    return FailoverDrillSnapshot(
        status=overall,
        generated_at=generated_at,
        scenario=scenario,
        components=tuple(components),
        health_report=health_report,
        failover_decision=decision,
        metadata=snapshot_metadata,
    )


def format_failover_drill_markdown(snapshot: FailoverDrillSnapshot) -> str:
    """Format a human-readable summary for runtime reports."""

    header = f"**Failover drill ({snapshot.scenario}) – status: {snapshot.status.value.upper()}**"
    return "\n".join([header, "", snapshot.to_markdown()])


__all__ = [
    "FailoverDrillComponent",
    "FailoverDrillSnapshot",
    "FailoverDrillStatus",
    "execute_failover_drill",
    "format_failover_drill_markdown",
]
