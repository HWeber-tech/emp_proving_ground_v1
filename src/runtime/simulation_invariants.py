"""Helpers for evaluating simulation invariant posture.

This module analyses :class:`PaperTradingSimulationReport` payloads produced by
``src.runtime.paper_simulation`` and surfaces a deterministic assessment of risk
invariant health.  Roadmap acceptance requires proving that longer simulations
(e.g. four hour rehearsals) complete without guardrail violations; the helpers
here centralise that logic so CLIs and tests share the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.runtime.paper_simulation import PaperTradingSimulationReport

__all__ = [
    "SimulationInvariantError",
    "SimulationInvariantAssessment",
    "assess_simulation_invariants",
    "ensure_zero_invariants",
]


class SimulationInvariantError(RuntimeError):
    """Raised when a simulation run fails invariant acceptance."""


@dataclass(frozen=True)
class SimulationInvariantAssessment:
    """Structured view of the invariant posture for a simulation run."""

    simulated_runtime_seconds: float
    tick_count: int
    guardrail_violations: int
    guardrail_near_misses: int
    last_guardrail_severity: str | None

    def ok(self) -> bool:
        """Return ``True`` when no guardrail violations were observed."""

        return self.guardrail_violations <= 0

    def as_dict(self) -> dict[str, object]:
        """Export the assessment as a plain mapping for JSON serialisation."""

        payload: dict[str, object] = {
            "simulated_runtime_seconds": self.simulated_runtime_seconds,
            "tick_count": self.tick_count,
            "guardrail_violations": self.guardrail_violations,
            "guardrail_near_misses": self.guardrail_near_misses,
        }
        if self.last_guardrail_severity is not None:
            payload["last_guardrail_severity"] = self.last_guardrail_severity
        return payload


def _extract_guardrail_counts(stats: Mapping[str, object] | None) -> tuple[int, int, str | None]:
    if not isinstance(stats, Mapping):
        return 0, 0, None

    violations = int(stats.get("guardrail_violations", 0) or 0)
    near_misses = int(stats.get("guardrail_near_misses", 0) or 0)

    severity: str | None = None
    incident = stats.get("last_guardrail_incident")
    if isinstance(incident, Mapping):
        raw = incident.get("severity") or incident.get("status")
        if isinstance(raw, str) and raw.strip():
            severity = raw.strip().lower()

    return violations, near_misses, severity


def assess_simulation_invariants(
    report: PaperTradingSimulationReport,
    *,
    simulated_tick_seconds: float = 0.5,
) -> SimulationInvariantAssessment:
    """Summarise invariant posture for ``report``.

    Parameters
    ----------
    report:
        Structured report produced by :func:`run_paper_trading_simulation`.
    simulated_tick_seconds:
        Assumed wall-clock coverage represented by each simulation tick.  The
        bootstrap runtime defaults to ``0.5`` seconds per tick; callers may
        override the value to match their configuration when ticks represent
        longer horizons.
    """

    if simulated_tick_seconds <= 0:
        raise ValueError("simulated_tick_seconds must be positive")

    stats = report.execution_stats if isinstance(report.execution_stats, Mapping) else None
    orders_submitted = int(stats.get("orders_submitted", 0) or 0) if stats else 0
    orders_executed = int(stats.get("orders_executed", 0) or 0) if stats else 0

    tick_count = max(int(report.decisions or 0), orders_submitted, orders_executed)
    simulated_runtime_seconds = tick_count * float(simulated_tick_seconds)

    violations, near_misses, severity = _extract_guardrail_counts(stats)

    return SimulationInvariantAssessment(
        simulated_runtime_seconds=simulated_runtime_seconds,
        tick_count=tick_count,
        guardrail_violations=violations,
        guardrail_near_misses=near_misses,
        last_guardrail_severity=severity,
    )


def ensure_zero_invariants(
    report: PaperTradingSimulationReport,
    *,
    required_runtime_seconds: float,
    simulated_tick_seconds: float = 0.5,
) -> SimulationInvariantAssessment:
    """Validate that ``report`` satisfies the zero-invariant acceptance gate.

    Raises
    ------
    SimulationInvariantError
        When the simulated runtime is below ``required_runtime_seconds`` or when
        guardrail violations are present.
    """

    if required_runtime_seconds <= 0:
        raise ValueError("required_runtime_seconds must be positive")

    assessment = assess_simulation_invariants(
        report,
        simulated_tick_seconds=simulated_tick_seconds,
    )

    if assessment.simulated_runtime_seconds < required_runtime_seconds:
        raise SimulationInvariantError(
            "Simulation runtime insufficient: required %.1fs, observed %.1fs"
            % (required_runtime_seconds, assessment.simulated_runtime_seconds)
        )

    if assessment.guardrail_violations > 0:
        raise SimulationInvariantError(
            "Guardrail violations observed: %d" % assessment.guardrail_violations
        )

    if (
        assessment.last_guardrail_severity
        and assessment.last_guardrail_severity.strip().lower() == "violation"
    ):
        raise SimulationInvariantError(
            "Last guardrail incident recorded a violation despite zero-count summary"
        )

    return assessment
