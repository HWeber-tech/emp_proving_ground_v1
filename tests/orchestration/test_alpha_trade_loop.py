from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.orchestration.alpha_trade_loop import AlphaTradeLoopOrchestrator
from src.trading.gating import DriftSentryGate
from src.understanding.belief import BeliefDistribution, BeliefState
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.router import (
    BeliefSnapshot,
    FastWeightAdapter,
    FeatureGate,
    PolicyTactic,
    RegimeState,
    UnderstandingRouter,
)


@pytest.mark.guardrail
def test_alpha_trade_loop_records_diary_and_forces_paper(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_strike",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={"requires_diary": True},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
            description="Aggressive alpha tactic",
            objectives=("alpha",),
            tags=("fast-weight", "momentum"),
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="beta_hold",
            base_weight=0.85,
            parameters={"mode": "beta"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    router.register_adapter(
        FastWeightAdapter(
            adapter_id="liquidity_rescue",
            tactic_id="alpha_strike",
            multiplier=1.25,
            rationale="Lean into alpha when liquidity is stressed",
            feature_gates=(FeatureGate(feature="liquidity_z", maximum=-0.1),),
            required_flags={"fast_weights_live": True},
        )
    )

    diary_path = tmp_path / "diary.json"
    diary_store = DecisionDiaryStore(diary_path, publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_strike",
        tactic_id="alpha_strike",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-integration",
    )
    release_manager = LedgerReleaseManager(ledger_store)

    drift_gate = DriftSentryGate()
    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    drift_snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        status=DriftSeverity.warn,
        dimensions={
            "liquidity": SensoryDimensionDrift(
                name="liquidity",
                current_signal=-0.35,
                baseline_signal=-0.05,
                delta=-0.30,
                current_confidence=0.5,
                baseline_confidence=0.7,
                confidence_delta=-0.2,
                severity=DriftSeverity.warn,
                samples=24,
            )
        },
        sample_window=32,
        metadata={"source": "pytest"},
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.82,
        features={"liquidity_z": -0.25, "momentum": 0.3},
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-balanced",
        regime_state=regime_state,
        features={"liquidity_z": -0.25, "momentum": 0.3},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=True,
        feature_flags={"fast_weights_live": True},
    )

    distribution = BeliefDistribution(
        mean=(0.0, 0.0),
        covariance=((1.0, 0.0), (0.0, 1.0)),
        strength=1.0,
        confidence=0.8,
        support=16,
        decay=0.1,
    )
    belief_state = BeliefState(
        belief_id="belief-balanced",
        version="1.0",
        symbol="EURUSD",
        generated_at=datetime(2024, 1, 1, 11, 55, tzinfo=UTC),
        features=("liquidity_z", "momentum"),
        prior=distribution,
        posterior=distribution,
        lineage={"source": "unit-test"},
        metadata={"window": "15m"},
    )

    trade_metadata = {"symbol": "EURUSD", "quantity": 25_000, "notional": 25_000.0}

    result = orchestrator.run_iteration(
        belief_snapshot,
        belief_state=belief_state,
        outcomes={"paper_pnl": 0.0},
        drift_snapshot=drift_snapshot,
        trade=trade_metadata,
        notes=["alpha-trade"],
        extra_metadata={"ticket": "alpha-loop"},
    )

    assert result.policy_id == "alpha_strike"
    assert result.decision.tactic_id == "alpha_strike"
    assert result.release_stage is PolicyLedgerStage.PAPER
    assert result.drift_decision.force_paper is True
    assert result.drift_decision.allowed is True
    assert result.diary_entry.metadata["release_stage"] == "paper"
    assert result.diary_entry.metadata["drift_decision"]["force_paper"] is True
    assert result.diary_entry.metadata["trade"]["symbol"] == "EURUSD"
    assert result.diary_entry.metadata["ticket"] == "alpha-loop"
    assert diary_store.entries()
    assert result.reflection.digest["total_decisions"] == 1
    assert result.metadata["force_paper"] is True
    assert result.metadata["release_stage"] == "paper"
