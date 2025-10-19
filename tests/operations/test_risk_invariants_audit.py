from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.operations.risk_invariants_audit import (
    AuditStatus,
    InvariantScenarioEvidence,
    audit_weekly_invariants,
    DEFAULT_RECENCY_THRESHOLD,
)
from src.runtime.paper_simulation import PaperTradingSimulationReport


def _make_report(
    *,
    decisions: int,
    orders_submitted: int,
    guardrail_violations: int,
    guardrail_near_misses: int,
    last_severity: str | None = None,
) -> PaperTradingSimulationReport:
    stats: dict[str, object] = {
        "orders_submitted": orders_submitted,
        "orders_executed": orders_submitted,
        "guardrail_violations": guardrail_violations,
        "guardrail_near_misses": guardrail_near_misses,
    }
    if last_severity is not None:
        stats["last_guardrail_incident"] = {"severity": last_severity}
    return PaperTradingSimulationReport(
        decisions=decisions,
        runtime_seconds=1.0,
        execution_stats=stats,
    )


def test_audit_reports_ok_when_all_scenarios_clean() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)
    executed_at = now - timedelta(days=1)

    evidence = [
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=executed_at,
        ),
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=executed_at,
        ),
        InvariantScenarioEvidence(
            scenario="bad_prints",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=executed_at,
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    assert audit.status is AuditStatus.ok
    assert not audit.missing_scenarios
    for snapshot in audit.scenarios:
        assert snapshot.status is AuditStatus.ok
        assert "completed" in snapshot.summary.lower()
        assert snapshot.assessment is not None
        assert snapshot.metadata["guardrail_violations"] == 0


def test_audit_warns_when_near_misses_present() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)
    executed_at = now - timedelta(days=1)

    evidence = [
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=3,
            ),
            executed_at=executed_at,
        ),
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=executed_at,
        ),
        InvariantScenarioEvidence(
            scenario="bad_prints",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=executed_at,
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    assert audit.status is AuditStatus.warn
    extreme_snapshot = next(s for s in audit.scenarios if s.scenario == "extreme_volatility")
    assert extreme_snapshot.status is AuditStatus.warn
    assert "near misses" in " ".join(extreme_snapshot.messages).lower()
    assert extreme_snapshot.metadata["guardrail_near_misses"] == 3


def test_audit_fails_when_scenario_missing() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)

    evidence = [
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    assert audit.status is AuditStatus.fail
    assert "bad_prints" in audit.missing_scenarios
    missing_snapshot = next(s for s in audit.scenarios if s.scenario == "bad_prints")
    assert missing_snapshot.status is AuditStatus.fail
    assert missing_snapshot.assessment is None


def test_audit_marks_stale_runs_warn() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)
    stale_time = now - (DEFAULT_RECENCY_THRESHOLD + timedelta(days=2))

    evidence = [
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=stale_time,
        ),
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
        InvariantScenarioEvidence(
            scenario="bad_prints",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    assert audit.status is AuditStatus.warn
    stale_snapshot = next(s for s in audit.scenarios if s.scenario == "extreme_volatility")
    assert stale_snapshot.status is AuditStatus.warn
    assert "stale" in " ".join(stale_snapshot.messages).lower()
    assert "stale_by_seconds" in stale_snapshot.metadata


def test_audit_flags_guardrail_violations_as_fail() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)

    evidence = [
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=2,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
        InvariantScenarioEvidence(
            scenario="bad_prints",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    assert audit.status is AuditStatus.fail
    extreme_snapshot = next(s for s in audit.scenarios if s.scenario == "extreme_volatility")
    assert extreme_snapshot.status is AuditStatus.fail
    assert "violations" in " ".join(extreme_snapshot.messages).lower()
    assert extreme_snapshot.metadata["guardrail_violations"] == 2


def test_latest_evidence_selected_for_duplicate_scenarios() -> None:
    now = datetime(2024, 6, 10, tzinfo=UTC)

    older = InvariantScenarioEvidence(
        scenario="extreme_volatility",
        report=_make_report(
            decisions=3600,
            orders_submitted=3600,
            guardrail_violations=0,
            guardrail_near_misses=0,
        ),
        executed_at=now - timedelta(days=5),
    )
    newer = InvariantScenarioEvidence(
        scenario="extreme_volatility",
        report=_make_report(
            decisions=7200,
            orders_submitted=7200,
            guardrail_violations=0,
            guardrail_near_misses=1,
        ),
        executed_at=now - timedelta(days=1),
    )
    evidence = [
        older,
        newer,
        InvariantScenarioEvidence(
            scenario="symbol_halt",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
        InvariantScenarioEvidence(
            scenario="bad_prints",
            report=_make_report(
                decisions=7200,
                orders_submitted=7200,
                guardrail_violations=0,
                guardrail_near_misses=0,
            ),
            executed_at=now - timedelta(days=1),
        ),
    ]

    audit = audit_weekly_invariants(evidence, now=now)

    extreme_snapshot = next(s for s in audit.scenarios if s.scenario == "extreme_volatility")
    assert extreme_snapshot.metadata["guardrail_near_misses"] == 1
    assert extreme_snapshot.metadata["tick_count"] == 7200


def test_invalid_evidence_raises_value_error() -> None:
    report = _make_report(
        decisions=7200,
        orders_submitted=7200,
        guardrail_violations=0,
        guardrail_near_misses=0,
    )
    with pytest.raises(ValueError):
        InvariantScenarioEvidence(
            scenario="",
            report=report,
            executed_at=datetime.now(tz=UTC),
        )
    with pytest.raises(ValueError):
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=report,
            executed_at=datetime.now(tz=UTC),
            required_runtime_seconds=0,
        )
    with pytest.raises(ValueError):
        InvariantScenarioEvidence(
            scenario="extreme_volatility",
            report=report,
            executed_at=datetime.now(tz=UTC),
            simulated_tick_seconds=0,
        )
