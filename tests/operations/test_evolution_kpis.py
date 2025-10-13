from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import src.operational.metrics as operational_metrics
from emp.core.findings_memory import TimeToCandidateBreach, TimeToCandidateStats
from src.governance.policy_ledger import PolicyLedgerRecord, PolicyLedgerStage
from src.operations.evolution_kpis import (
    EvolutionKpiStatus,
    evaluate_evolution_kpis,
)


def _stub_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    functions = [
        "set_evolution_time_to_candidate_stat",
        "set_evolution_time_to_candidate_total",
        "set_evolution_time_to_candidate_breaches",
        "set_evolution_promotion_counts",
        "set_evolution_promotion_transitions",
        "set_evolution_promotion_rate",
        "set_evolution_budget_usage",
        "set_evolution_budget_blocked",
        "set_evolution_budget_forced",
        "set_evolution_budget_samples",
        "set_evolution_rollback_latency",
        "set_evolution_rollback_events",
    ]

    for name in functions:
        monkeypatch.setattr(operational_metrics, name, lambda *args, **kwargs: None)


def _ledger_record(now: datetime) -> PolicyLedgerRecord:
    history = (
        {
            "prior_stage": None,
            "next_stage": PolicyLedgerStage.EXPERIMENT.value,
            "applied_at": (now - timedelta(days=2)).isoformat(),
        },
        {
            "prior_stage": PolicyLedgerStage.EXPERIMENT.value,
            "next_stage": PolicyLedgerStage.PAPER.value,
            "applied_at": (now - timedelta(days=1, hours=6)).isoformat(),
        },
        {
            "prior_stage": PolicyLedgerStage.PAPER.value,
            "next_stage": PolicyLedgerStage.PILOT.value,
            "applied_at": (now - timedelta(hours=10)).isoformat(),
        },
        {
            "prior_stage": PolicyLedgerStage.PILOT.value,
            "next_stage": PolicyLedgerStage.PAPER.value,
            "applied_at": (now - timedelta(hours=2)).isoformat(),
        },
    )

    return PolicyLedgerRecord(
        policy_id="alpha.paper",
        tactic_id="alpha.paper",
        stage=PolicyLedgerStage.PAPER,
        approvals=(),
        evidence_id=None,
        threshold_overrides={},
        policy_delta=None,
        metadata={},
        created_at=now - timedelta(days=2),
        updated_at=now,
        history=history,
    )


def _loop_results() -> list[object]:
    metadata = {
        "budget_after": {
            "exploration_share": 0.12,
            "max_fraction": 0.2,
            "blocked_attempts": 1,
            "forced_decisions": 0,
        }
    }

    class _Decision:
        def __init__(self, exploration_metadata: dict[str, object]) -> None:
            self.exploration_metadata = exploration_metadata

    class _LoopResult:
        def __init__(self, exploration_metadata: dict[str, object]) -> None:
            self.decision = _Decision(exploration_metadata)

    return [_LoopResult(metadata)]


def test_evaluate_evolution_kpis_aggregates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_metrics(monkeypatch)

    now = datetime.now(timezone.utc)
    stats = TimeToCandidateStats(
        count=5,
        average_hours=18.0,
        median_hours=12.0,
        p90_hours=26.0,
        max_hours=36.0,
        threshold_hours=24.0,
        sla_met=False,
        breaches=(
            TimeToCandidateBreach(
                id=101,
                stage="tested",
                created_at="2025-01-01T00:00:00Z",
                tested_at="2025-01-02T06:00:00Z",
                hours=30.0,
            ),
        ),
    )

    record = _ledger_record(now)
    snapshot = evaluate_evolution_kpis(
        time_to_candidate=stats,
        ledger_records=[record],
        loop_results=_loop_results(),
        now=now,
    )

    assert snapshot.status is EvolutionKpiStatus.fail
    assert snapshot.time_to_candidate is not None
    assert snapshot.time_to_candidate.count == 5
    assert snapshot.time_to_candidate.sla_met is False
    assert snapshot.promotion is not None
    assert snapshot.promotion.promotions == 2
    assert snapshot.promotion.demotions == 1
    assert snapshot.budget is not None
    assert snapshot.budget.samples == 1
    assert snapshot.rollback is not None
    assert snapshot.rollback.samples == 1

    payload = snapshot.as_dict()
    assert payload["status"] == EvolutionKpiStatus.fail.value
    assert "time_to_candidate" in payload
    assert payload["promotion"]["promotion_rate"] > 0


def test_evaluate_evolution_kpis_handles_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_metrics(monkeypatch)

    snapshot = evaluate_evolution_kpis()

    assert snapshot.status is EvolutionKpiStatus.ok
    assert snapshot.time_to_candidate is None
    assert snapshot.promotion is None
    assert snapshot.budget is None
    assert snapshot.rollback is None
