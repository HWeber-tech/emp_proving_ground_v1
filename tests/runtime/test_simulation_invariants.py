from __future__ import annotations

import pytest

from src.runtime.paper_simulation import PaperTradingSimulationReport
from src.runtime.simulation_invariants import (
    SimulationInvariantAssessment,
    SimulationInvariantError,
    assess_simulation_invariants,
    ensure_zero_invariants,
)


def _make_report(
    *,
    decisions: int,
    orders_submitted: int,
    guardrail_violations: int,
    guardrail_near_misses: int,
    last_severity: str | None = None,
) -> PaperTradingSimulationReport:
    execution_stats: dict[str, object] = {
        "orders_submitted": orders_submitted,
        "orders_executed": orders_submitted,
        "guardrail_violations": guardrail_violations,
        "guardrail_near_misses": guardrail_near_misses,
    }
    if last_severity is not None:
        execution_stats["last_guardrail_incident"] = {"severity": last_severity}
    return PaperTradingSimulationReport(
        decisions=decisions,
        runtime_seconds=1.0,
        execution_stats=execution_stats,
    )


def test_assess_simulation_invariants_aggregates_counts() -> None:
    report = _make_report(
        decisions=28800,
        orders_submitted=28800,
        guardrail_violations=0,
        guardrail_near_misses=12,
    )

    assessment = assess_simulation_invariants(report, simulated_tick_seconds=0.5)

    assert isinstance(assessment, SimulationInvariantAssessment)
    assert assessment.tick_count == 28800
    assert assessment.simulated_runtime_seconds == pytest.approx(14400.0)
    assert assessment.guardrail_violations == 0
    assert assessment.guardrail_near_misses == 12
    assert assessment.ok()


def test_ensure_zero_invariants_passes_for_four_hour_run() -> None:
    report = _make_report(
        decisions=28800,
        orders_submitted=28800,
        guardrail_violations=0,
        guardrail_near_misses=8,
    )

    assessment = ensure_zero_invariants(
        report,
        required_runtime_seconds=4 * 3600,
        simulated_tick_seconds=0.5,
    )

    assert assessment.ok()


def test_ensure_zero_invariants_requires_minimum_runtime() -> None:
    report = _make_report(
        decisions=1000,
        orders_submitted=1000,
        guardrail_violations=0,
        guardrail_near_misses=4,
    )

    with pytest.raises(SimulationInvariantError) as exc:
        ensure_zero_invariants(
            report,
            required_runtime_seconds=4 * 3600,
            simulated_tick_seconds=0.5,
        )

    assert "insufficient" in str(exc.value).lower()


def test_ensure_zero_invariants_rejects_guardrail_violations() -> None:
    report = _make_report(
        decisions=28800,
        orders_submitted=28800,
        guardrail_violations=1,
        guardrail_near_misses=0,
    )

    with pytest.raises(SimulationInvariantError) as exc:
        ensure_zero_invariants(
            report,
            required_runtime_seconds=4 * 3600,
            simulated_tick_seconds=0.5,
        )

    assert "guardrail" in str(exc.value).lower()


def test_ensure_zero_invariants_flags_violation_severity() -> None:
    report = _make_report(
        decisions=28800,
        orders_submitted=28800,
        guardrail_violations=0,
        guardrail_near_misses=5,
        last_severity="violation",
    )

    with pytest.raises(SimulationInvariantError):
        ensure_zero_invariants(
            report,
            required_runtime_seconds=4 * 3600,
            simulated_tick_seconds=0.5,
        )
