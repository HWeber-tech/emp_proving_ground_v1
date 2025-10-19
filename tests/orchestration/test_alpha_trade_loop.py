from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.evolution.feature_flags import EvolutionFeatureFlags
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    ComplianceEventType,
    ComplianceSeverity,
)
from src.trading.gating import DriftSentryGate
from src.thinking.adaptation.evolution_manager import (
    EvolutionManager,
    ManagedStrategyConfig,
    StrategyVariant,
)
from src.thinking.adaptation.policy_router import FastWeightExperiment
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
from tests.util import promotion_checklist_metadata


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
    assert result.metadata["release_stage_sources"] == {
        "policy": "paper",
        "tactic": "paper",
    }
    guardrails = result.decision.guardrails
    assert guardrails["force_paper"] is True
    assert guardrails["release_stage"] == "paper"
    assert guardrails["governance_release_stage"] == "paper"
    assert guardrails["governance_policy_stage"] == "paper"
    assert guardrails["governance_tactic_stage"] == "paper"
    assert (
        guardrails["governance_release_stage_gate"]
        == "release_stage_paper_requires_paper_execution"
    )


@pytest.mark.guardrail
def test_alpha_trade_loop_counterfactual_guardrail_forces_paper(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_live",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    router.policy_router.register_experiment(
        FastWeightExperiment(
            experiment_id="live_burst",
            tactic_id="alpha_live",
            delta=1.0,
            rationale="double weight for guardrail test",
            min_confidence=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_live",
        tactic_id="alpha_live",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-live",
        metadata=promotion_checklist_metadata(),
    )
    release_manager = LedgerReleaseManager(ledger_store)

    drift_gate = DriftSentryGate()
    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.9,
        features={"liquidity_z": 0.1, "momentum": 0.4},
        timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-live",
        regime_state=regime_state,
        features={"liquidity_z": 0.1, "momentum": 0.4},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=True,
        feature_flags={"fast_weights_live": True},
    )

    result = orchestrator.run_iteration(
        belief_snapshot,
        policy_id="alpha_live",
        trade={"symbol": "EURUSD", "quantity": 10_000, "price": 1.25},
    )

    assert result.release_stage is PolicyLedgerStage.LIMITED_LIVE
    guardrails = result.decision.guardrails
    assert guardrails["force_paper"] is True
    counterfactual = guardrails.get("counterfactual_guardrail")
    assert isinstance(counterfactual, dict)
    assert counterfactual.get("breached") is True
    assert counterfactual.get("reason") == "counterfactual_guardrail_delta_exceeded"
    assert counterfactual.get("action") == "force_paper"
    assert counterfactual.get("max_relative_delta") == pytest.approx(0.20)
    assert result.metadata["force_paper"] is True
    assert diary_store.entries(), "expected decision diary entry"


@pytest.mark.guardrail
def test_alpha_trade_loop_counterfactual_guardrail_respects_override(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_live",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    router.policy_router.register_experiment(
        FastWeightExperiment(
            experiment_id="live_burst",
            tactic_id="alpha_live",
            delta=1.0,
            rationale="double weight for guardrail test",
            min_confidence=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_live",
        tactic_id="alpha_live",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-live",
        threshold_overrides={"counterfactual_relative_delta_limit": 1.5},
        metadata=promotion_checklist_metadata(),
    )
    release_manager = LedgerReleaseManager(ledger_store)

    drift_gate = DriftSentryGate()
    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.9,
        features={"liquidity_z": 0.1, "momentum": 0.4},
        timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-live",
        regime_state=regime_state,
        features={"liquidity_z": 0.1, "momentum": 0.4},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=True,
        feature_flags={"fast_weights_live": True},
    )

    result = orchestrator.run_iteration(
        belief_snapshot,
        policy_id="alpha_live",
        trade={"symbol": "EURUSD", "quantity": 10_000, "price": 1.25},
    )

    guardrails = result.decision.guardrails
    counterfactual = guardrails.get("counterfactual_guardrail")
    assert isinstance(counterfactual, dict)
    assert counterfactual.get("breached") is False
    assert counterfactual.get("max_relative_delta") == pytest.approx(1.5)
    assert guardrails.get("force_paper") is False
    assert result.metadata["force_paper"] is False


def test_alpha_trade_loop_paper_stage_forces_paper_without_warn(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_shadow",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_shadow",
        tactic_id="alpha_shadow",
        stage=PolicyLedgerStage.PAPER,
        approvals=(),
        evidence_id="dd-shadow",
    )
    release_manager = LedgerReleaseManager(ledger_store)

    drift_gate = DriftSentryGate()
    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.9,
        features={"momentum": 0.1},
        timestamp=datetime(2024, 1, 2, 9, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-shadow",
        regime_state=regime_state,
        features={"momentum": 0.1},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=False,
        feature_flags={},
    )

    trade_metadata = {"symbol": "EURUSD", "quantity": 10_000, "notional": 10_000.0}

    result = orchestrator.run_iteration(
        belief_snapshot,
        policy_id="alpha_shadow",
        trade=trade_metadata,
        outcomes={"paper_pnl": 0.0},
    )

    assert result.release_stage is PolicyLedgerStage.PAPER
    assert result.drift_decision.severity is DriftSeverity.normal
    assert result.drift_decision.force_paper is True
    assert (
        result.drift_decision.reason
        == "release_stage_paper_requires_paper_execution"
    )
    assert result.metadata["force_paper"] is True
    assert result.metadata["release_stage"] == "paper"
    assert result.metadata["release_stage_sources"] == {
        "policy": "paper",
        "tactic": "paper",
    }
    assert result.diary_entry.metadata["drift_decision"]["force_paper"] is True
    assert (
        result.diary_entry.metadata["drift_decision"]["reason"]
        == "release_stage_paper_requires_paper_execution"
    )


def test_alpha_trade_loop_enforces_more_conservative_stage(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_concept",
            base_weight=1.0,
            parameters={"mode": "concept"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="aggregated_policy",
        tactic_id="alpha_concept",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "product"),
        evidence_id="pilot-approved",
    )

    release_manager = LedgerReleaseManager(ledger_store)
    drift_gate = DriftSentryGate()
    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.88,
        features={"momentum": 0.15},
        timestamp=datetime(2024, 2, 1, 10, 30, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-concept",
        regime_state=regime_state,
        features={"momentum": 0.15},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=False,
        feature_flags={},
    )

    trade_metadata = {"symbol": "EURUSD", "quantity": 5_000, "notional": 5_000.0}

    result = orchestrator.run_iteration(
        belief_snapshot,
        policy_id="aggregated_policy",
        trade=trade_metadata,
        outcomes={"paper_pnl": 0.0},
    )

    assert result.release_stage is PolicyLedgerStage.EXPERIMENT
    assert result.drift_decision.force_paper is True
    assert (
        result.drift_decision.reason
        == "release_stage_experiment_requires_paper_or_better"
    )
    assert result.metadata["release_stage_sources"] == {
        "policy": "pilot",
        "tactic": "experiment",
    }
    assert result.metadata["release_stage"] == "experiment"
    assert result.metadata["force_paper"] is True
    assert (
        result.diary_entry.metadata["drift_decision"]["reason"]
        == "release_stage_experiment_requires_paper_or_better"
    )
    guardrails = result.decision.guardrails
    assert guardrails["force_paper"] is True
    assert guardrails["release_stage"] == "experiment"
    assert guardrails["governance_release_stage"] == "experiment"
    assert guardrails["governance_policy_stage"] == "pilot"
    assert guardrails["governance_tactic_stage"] == "experiment"
    assert (
        guardrails["governance_release_stage_gate"]
        == "release_stage_experiment_requires_paper_or_better"
    )


def test_alpha_trade_loop_live_stage_allows_live_execution(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_live",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_live",
        tactic_id="alpha_live",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "compliance"),
        evidence_id="live-ready",
        metadata=promotion_checklist_metadata(),
    )

    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=DriftSentryGate(),
        release_manager=LedgerReleaseManager(ledger_store),
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.95,
        features={"momentum": 0.2},
        timestamp=datetime(2024, 4, 1, 9, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-live",
        regime_state=regime_state,
        features={"momentum": 0.2},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=True,
        feature_flags={},
    )

    result = orchestrator.run_iteration(
        belief_snapshot,
        trade={"symbol": "EURUSD", "quantity": 15_000, "notional": 15_000.0},
        outcomes={"paper_pnl": 100.0},
    )

    assert result.release_stage is PolicyLedgerStage.LIMITED_LIVE
    assert result.drift_decision.force_paper is False
    assert result.metadata["force_paper"] is False
    guardrails = result.decision.guardrails
    assert guardrails["force_paper"] is False
    assert guardrails["release_stage"] == "limited_live"
    assert guardrails["governance_release_stage"] == "limited_live"
    assert guardrails["governance_policy_stage"] == "limited_live"
    assert guardrails["governance_tactic_stage"] == "limited_live"
    assert "governance_release_stage_gate" not in guardrails


def test_alpha_trade_loop_emits_stage_gate_compliance_event(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_shadow",
            base_weight=1.0,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_shadow",
        tactic_id="alpha_shadow",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="stage-gate",
    )

    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=DriftSentryGate(),
        release_manager=LedgerReleaseManager(ledger_store),
    )

    regime_state = RegimeState(
        regime="balanced",
        confidence=0.7,
        features={"momentum": 0.05},
        timestamp=datetime(2024, 3, 1, 9, 0, tzinfo=UTC),
    )
    belief_snapshot = BeliefSnapshot(
        belief_id="belief-shadow",
        regime_state=regime_state,
        features={"momentum": 0.05},
        metadata={"symbol": "EURUSD"},
        fast_weights_enabled=False,
        feature_flags={},
    )

    result = orchestrator.run_iteration(
        belief_snapshot,
        policy_id="alpha_shadow",
        outcomes={"paper_pnl": 0.0},
        trade={"symbol": "EURUSD", "quantity": 10_000, "notional": 10_000.0},
    )

    events = result.compliance_events
    assert len(events) == 1
    event = events[0]
    assert event.event_type is ComplianceEventType.governance_action
    assert event.severity is ComplianceSeverity.info
    assert (
        event.summary
        == "Policy alpha_shadow forced to paper by governance stage paper (drift severity normal)"
    )
    event_metadata = dict(event.metadata)
    assert event_metadata["release_stage"] == "paper"
    assert event_metadata["action"] == "force_paper"
    assert (
        event_metadata["release_stage_gate"]
        == "release_stage_paper_requires_paper_execution"
    )


def test_alpha_trade_loop_annotates_evolution_metadata(tmp_path: Path) -> None:
    router = UnderstandingRouter()
    base_tactic = PolicyTactic(
        tactic_id="alpha_loop",
        base_weight=1.0,
        parameters={"mode": "alpha"},
        guardrails={"requires_diary": True},
        regime_bias={"balanced": 1.0},
        confidence_sensitivity=0.0,
        description="Alpha loop baseline",
        tags=("alpha",),
    )
    router.register_tactic(base_tactic)

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    ledger_store.upsert(
        policy_id="alpha_loop",
        tactic_id="alpha_loop",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-loop",
    )
    release_manager = LedgerReleaseManager(ledger_store)

    variant = StrategyVariant(
        variant_id="alpha_loop_trial",
        tactic=PolicyTactic(
            tactic_id="alpha_loop_trial",
            base_weight=0.85,
            parameters={"mode": "alpha", "regime_bias": "storm"},
            guardrails={"force_paper": True},
            description="Exploratory alpha variant",
        ),
        base_tactic_id="alpha_loop",
        rationale="Register exploratory alpha variant after losses",
        trial_weight_multiplier=1.1,
    )

    evolution_manager = EvolutionManager(
        policy_router=router.policy_router,
        strategies=(
            ManagedStrategyConfig(
                base_tactic_id="alpha_loop",
                fallback_variants=(variant,),
                degrade_multiplier=0.6,
            ),
        ),
        window_size=2,
        win_rate_threshold=0.5,
        feature_flags=EvolutionFeatureFlags(env={"EVOLUTION_ENABLE_ADAPTIVE_RUNS": "1"}),
    )

    orchestrator = AlphaTradeLoopOrchestrator(
        router=router,
        diary_store=diary_store,
        drift_gate=DriftSentryGate(),
        release_manager=release_manager,
        evolution_manager=evolution_manager,
    )

    base_time = datetime(2024, 4, 1, 9, 0, tzinfo=UTC)
    distribution = BeliefDistribution(
        mean=(0.0,),
        covariance=((1.0,),),
        strength=1.0,
        confidence=0.6,
        support=8,
        decay=0.1,
    )
    second_result = None
    for index in range(2):
        regime_state = RegimeState(
            regime="balanced",
            confidence=0.8,
            features={"momentum": -0.3},
            timestamp=base_time + timedelta(minutes=index),
        )
        belief_state = BeliefState(
            belief_id=f"belief-{index}",
            version="1.0",
            symbol="EURUSD",
            generated_at=regime_state.timestamp,
            features=("momentum",),
            prior=distribution,
            posterior=distribution,
            lineage={"source": "unit-test"},
        )
        belief_snapshot = BeliefSnapshot(
            belief_id=belief_state.belief_id,
            regime_state=regime_state,
            features={"momentum": -0.3},
            metadata={"iteration": index},
            fast_weights_enabled=True,
            feature_flags={},
        )

        result = orchestrator.run_iteration(
            belief_snapshot,
            belief_state=belief_state,
            policy_id="alpha_loop",
            outcomes={"paper_pnl": -50.0},
            notes=(f"iteration-{index}",),
        )

        if index == 0:
            assert "evolution" not in result.metadata
        else:
            second_result = result

    assert second_result is not None
    evolution_meta = second_result.metadata.get("evolution")
    assert evolution_meta is not None
    assert evolution_meta["base_tactic_id"] == "alpha_loop"
    assert evolution_meta["observations"] == 2
    actions = evolution_meta["actions"]
    action_types = {action["action"] for action in actions}
    assert {"register_variant", "degrade_base"} <= action_types

    diary_evolution = second_result.diary_entry.metadata.get("evolution")
    assert diary_evolution == evolution_meta

    router_tactics = router.policy_router.tactics()
    assert router_tactics["alpha_loop"].base_weight == pytest.approx(0.6)
    assert router_tactics["alpha_loop_trial"].base_weight == pytest.approx(0.935)
