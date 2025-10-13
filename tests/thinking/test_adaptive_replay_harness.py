from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.governance.adaptive_gate import AdaptiveGovernanceGate
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.thinking.adaptation.policy_router import PolicyTactic
from src.thinking.adaptation.replay_harness import (
    StageDecision,
    StageThresholds,
    TacticReplayHarness,
)
from src.evolution.evaluation.recorded_replay import RecordedSensorySnapshot


def _build_positive_snapshots() -> tuple[RecordedSensorySnapshot, ...]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    step = timedelta(minutes=5)
    return (
        RecordedSensorySnapshot(base, 100.0, 0.5, 0.9),
        RecordedSensorySnapshot(base + step, 102.0, 0.6, 0.92),
        RecordedSensorySnapshot(base + 2 * step, 104.0, 0.1, 0.91),
        RecordedSensorySnapshot(base + 3 * step, 105.0, -0.5, 0.9),
        RecordedSensorySnapshot(base + 4 * step, 103.0, -0.6, 0.9),
        RecordedSensorySnapshot(base + 5 * step, 101.0, -0.1, 0.88),
        RecordedSensorySnapshot(base + 6 * step, 101.5, 0.0, 0.9),
    )


def _build_negative_snapshots() -> tuple[RecordedSensorySnapshot, ...]:
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    step = timedelta(minutes=5)
    return (
        RecordedSensorySnapshot(base, 100.0, 0.5, 0.9),
        RecordedSensorySnapshot(base + step, 98.0, 0.0, 0.88),
        RecordedSensorySnapshot(base + 2 * step, 97.5, -0.6, 0.9),
        RecordedSensorySnapshot(base + 3 * step, 98.5, -0.1, 0.88),
        RecordedSensorySnapshot(base + 4 * step, 98.7, 0.2, 0.9),
    )


def _stage_thresholds() -> dict[PolicyLedgerStage, StageThresholds]:
    return {
        PolicyLedgerStage.EXPERIMENT: StageThresholds(
            promote_total_return=0.01,
            promote_win_rate=0.5,
            promote_sharpe=0.1,
            promote_max_drawdown=0.3,
            demote_total_return=-0.05,
            demote_win_rate=0.4,
            demote_sharpe=-0.2,
            demote_max_drawdown=0.5,
            min_trades=2,
            min_confidence=0.5,
        ),
        PolicyLedgerStage.PAPER: StageThresholds(
            promote_total_return=0.05,
            promote_win_rate=0.55,
            promote_sharpe=0.4,
            promote_max_drawdown=0.25,
            demote_total_return=-0.01,
            demote_win_rate=0.48,
            demote_sharpe=0.05,
            demote_max_drawdown=0.2,
            min_trades=1,
            min_confidence=0.55,
        ),
    }


def _tactic() -> PolicyTactic:
    return PolicyTactic(
        tactic_id="momentum_long_short",
        base_weight=1.0,
        parameters={
            "entry_threshold": 0.4,
            "exit_threshold": 0.2,
            "risk_fraction": 0.2,
            "min_confidence": 0.5,
        },
        guardrails={},
        regime_bias={"balanced": 1.0},
        confidence_sensitivity=0.0,
    )


def test_replay_harness_promotes_and_updates_ledger(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger_promo"
    ledger_dir.mkdir()
    ledger_store = PolicyLedgerStore(ledger_dir / "policy.json")
    release_manager = LedgerReleaseManager(ledger_store)
    harness = TacticReplayHarness(
        snapshots=_build_positive_snapshots(),
        release_manager=release_manager,
        stage_thresholds=_stage_thresholds(),
    )

    result = harness.evaluate_tactic(_tactic())

    assert result.decision is StageDecision.promote
    assert result.current_stage is PolicyLedgerStage.EXPERIMENT
    assert result.target_stage is PolicyLedgerStage.PAPER

    gate = AdaptiveGovernanceGate(release_manager)
    record = gate.apply_decision(result, evaluation_id="promo-run", approvals=("qa",))

    assert record is not None
    assert release_manager.resolve_stage(result.policy_id) is PolicyLedgerStage.PAPER
    assert record.metadata.get("evaluation_id") == "promo-run"


def test_replay_harness_demotes_on_poor_performance(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger_demote"
    ledger_dir.mkdir()
    ledger_store = PolicyLedgerStore(ledger_dir / "policy.json")
    ledger_store.upsert(
        policy_id="momentum_long_short",
        tactic_id="momentum_long_short",
        stage=PolicyLedgerStage.PAPER,
        approvals=("qa",),
        evidence_id="dd-1",
    )
    release_manager = LedgerReleaseManager(ledger_store)
    thresholds = _stage_thresholds()
    harness = TacticReplayHarness(
        snapshots=_build_negative_snapshots(),
        release_manager=release_manager,
        stage_thresholds=thresholds,
    )

    result = harness.evaluate_tactic(_tactic())

    assert result.current_stage is PolicyLedgerStage.PAPER
    assert result.decision is StageDecision.demote
    assert result.target_stage is PolicyLedgerStage.EXPERIMENT

    gate = AdaptiveGovernanceGate(release_manager)
    record = gate.apply_decision(result, evaluation_id="demotion-run")

    assert record is not None
    assert release_manager.resolve_stage(result.policy_id) is PolicyLedgerStage.EXPERIMENT
    assert record.metadata.get("decision") == StageDecision.demote.value


def test_replay_harness_maintains_when_insufficient_trades(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger_neutral"
    ledger_dir.mkdir()
    ledger_store = PolicyLedgerStore(ledger_dir / "policy.json")
    release_manager = LedgerReleaseManager(ledger_store)
    thresholds = _stage_thresholds()
    harness = TacticReplayHarness(
        snapshots=[
            RecordedSensorySnapshot(
                datetime(2024, 3, 1, 0, 0, tzinfo=timezone.utc),
                100.0,
                0.1,
                0.8,
            ),
            RecordedSensorySnapshot(
                datetime(2024, 3, 1, 0, 5, tzinfo=timezone.utc),
                100.5,
                0.15,
                0.82,
            ),
        ],
        release_manager=release_manager,
        stage_thresholds=thresholds,
    )

    result = harness.evaluate_tactic(_tactic())

    assert result.decision is StageDecision.maintain
    assert result.target_stage is PolicyLedgerStage.EXPERIMENT
