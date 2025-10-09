from __future__ import annotations

import pytest

from src.evolution.feature_flags import EvolutionFeatureFlags
from src.governance.policy_ledger import PolicyLedgerStage
from src.thinking.adaptation.evolution_manager import (
    CatalogueVariantRequest,
    EvolutionManager,
    ManagedStrategyConfig,
    StrategyVariant,
)
from src.thinking.adaptation.policy_router import PolicyDecision, PolicyRouter, PolicyTactic
from src.trading.strategies.catalog_loader import (
    BaselineDefinition,
    StrategyCatalog,
    StrategyDefinition,
)


def _decision(tactic_id: str) -> PolicyDecision:
    return PolicyDecision(
        tactic_id=tactic_id,
        parameters={},
        selected_weight=1.0,
        guardrails={},
        rationale="",
        experiments_applied=(),
        reflection_summary={},
        weight_breakdown={},
    )


def test_evolution_manager_registers_variant_on_losses() -> None:
    router = PolicyRouter()
    base_tactic = PolicyTactic(
        tactic_id="momentum_core",
        base_weight=1.0,
        parameters={"mode": "base"},
    )
    router.register_tactic(base_tactic)

    variant = StrategyVariant(
        variant_id="momentum_core_explore",
        tactic=PolicyTactic(
            tactic_id="momentum_core_explore",
            base_weight=0.9,
            parameters={"mode": "explore"},
            description="Exploratory momentum variant",
        ),
        base_tactic_id="momentum_core",
        rationale="Promote exploratory variant after losses",
        trial_weight_multiplier=1.1,
    )

    manager = EvolutionManager(
        policy_router=router,
        strategies=(
            ManagedStrategyConfig(
                base_tactic_id="momentum_core",
                fallback_variants=(variant,),
                degrade_multiplier=0.5,
            ),
        ),
        window_size=3,
        win_rate_threshold=0.34,
        feature_flags=EvolutionFeatureFlags(env={"EVOLUTION_ENABLE_ADAPTIVE_RUNS": "1"}),
    )

    decision = _decision("momentum_core")
    for _ in range(3):
        manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -25.0},
        )

    tactics = router.tactics()
    assert "momentum_core_explore" in tactics
    assert tactics["momentum_core"].base_weight == pytest.approx(0.5)
    assert tactics["momentum_core_explore"].base_weight == pytest.approx(0.99)


def test_evolution_manager_respects_feature_flag() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="mean_rev", base_weight=1.0))

    manager = EvolutionManager(
        policy_router=router,
        strategies=(ManagedStrategyConfig(base_tactic_id="mean_rev"),),
        window_size=2,
        win_rate_threshold=0.5,
        feature_flags=EvolutionFeatureFlags(env={}),
    )

    decision = _decision("mean_rev")
    for _ in range(3):
        manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -5.0},
        )

    assert router.tactics()["mean_rev"].base_weight == pytest.approx(1.0)


def test_evolution_manager_ignores_non_paper_stage() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="breakout", base_weight=1.0))

    manager = EvolutionManager(
        policy_router=router,
        strategies=(ManagedStrategyConfig(base_tactic_id="breakout"),),
        window_size=2,
        win_rate_threshold=0.5,
        feature_flags=EvolutionFeatureFlags(env={"EVOLUTION_ENABLE_ADAPTIVE_RUNS": "1"}),
    )

    decision = _decision("breakout")
    for _ in range(2):
        manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PILOT,
            outcomes={"paper_pnl": -12.0},
        )

    assert router.tactics()["breakout"].base_weight == pytest.approx(1.0)


def test_evolution_manager_registers_catalogue_variant() -> None:
    router = PolicyRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="momentum_core",
            base_weight=1.0,
            parameters={"mode": "core"},
            description="Core momentum configuration",
        )
    )

    catalogue = StrategyCatalog(
        version="test",
        default_capital=1_000_000.0,
        definitions=(
            StrategyDefinition(
                key="momentum_trial",
                identifier="momentum_trial_v1",
                name="Momentum Trial",
                class_name="MomentumStrategy",
                enabled=True,
                capital=900_000.0,
                parameters={"lookback": 14, "entry_threshold": 0.6},
                symbols=("EURUSD",),
                tags=("momentum", "trial"),
                description="Catalogue supplied momentum variant",
            ),
        ),
        baseline=BaselineDefinition(
            identifier="baseline",
            short_window=5,
            long_window=18,
            risk_fraction=0.25,
        ),
        description="Unit-test catalogue",
    )

    manager = EvolutionManager(
        policy_router=router,
        strategies=(
            ManagedStrategyConfig(
                base_tactic_id="momentum_core",
                degrade_multiplier=0.5,
                catalogue_variants=(
                    CatalogueVariantRequest(
                        strategy_id="momentum_trial_v1",
                        weight_multiplier=1.05,
                        rationale="Trial catalogue momentum variant",
                    ),
                ),
            ),
        ),
        window_size=3,
        win_rate_threshold=0.34,
        feature_flags=EvolutionFeatureFlags(env={"EVOLUTION_ENABLE_ADAPTIVE_RUNS": "1"}),
        catalogue=catalogue,
    )

    decision = _decision("momentum_core")
    for _ in range(3):
        manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -25.0},
            metadata={"scenario": "losing-streak"},
        )

    tactics = router.tactics()
    assert "momentum_trial_v1__trial" in tactics
    trial = tactics["momentum_trial_v1__trial"]
    assert trial.base_weight == pytest.approx(0.945)  # 0.9 * 1.05
    assert trial.parameters["catalogue_identifier"] == "momentum_trial_v1"
    assert trial.parameters["catalogue_version"] == "test"
    assert trial.guardrails["force_paper"] is True
    assert trial.tags == ("momentum", "trial")
    assert router.tactics()["momentum_core"].base_weight == pytest.approx(0.5)
