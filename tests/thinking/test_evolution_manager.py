from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.evolution.feature_flags import EvolutionFeatureFlags
from src.governance.policy_ledger import PolicyLedgerStage
from src.thinking.adaptation.evolution_manager import (
    CatalogueVariantRequest,
    EvolutionAdaptationResult,
    EvolutionManager,
    ManagedStrategyConfig,
    ParameterMutation,
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
        decision_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
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
    result: EvolutionAdaptationResult | None = None
    for _ in range(3):
        result = manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -25.0},
        )

    assert isinstance(result, EvolutionAdaptationResult)
    tactics = router.tactics()
    assert "momentum_core_explore" in tactics
    assert tactics["momentum_core"].base_weight == pytest.approx(0.5)
    assert tactics["momentum_core_explore"].base_weight == pytest.approx(0.99)

    payload = result.as_dict()
    assert payload["base_tactic_id"] == "momentum_core"
    assert payload["observations"] == 3
    assert payload["win_rate"] == pytest.approx(0.0)
    actions = payload["actions"]
    action_types = {action["action"] for action in actions}
    assert {"register_variant", "degrade_base"}.issubset(action_types)


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
    result = None
    for _ in range(3):
        result = manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -5.0},
        )

    assert result is None
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
    result = None
    for _ in range(2):
        result = manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PILOT,
            outcomes={"paper_pnl": -12.0},
        )

    assert result is None
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
    result: EvolutionAdaptationResult | None = None
    for _ in range(3):
        result = manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_pnl": -25.0},
            metadata={"scenario": "losing-streak"},
        )

    assert isinstance(result, EvolutionAdaptationResult)
    tactics = router.tactics()
    assert "momentum_trial_v1__trial" in tactics
    trial = tactics["momentum_trial_v1__trial"]
    assert trial.base_weight == pytest.approx(0.945)  # 0.9 * 1.05
    assert trial.parameters["catalogue_identifier"] == "momentum_trial_v1"
    assert trial.parameters["catalogue_version"] == "test"
    assert trial.guardrails["force_paper"] is True
    assert trial.tags == ("momentum", "trial")
    assert router.tactics()["momentum_core"].base_weight == pytest.approx(0.5)

    payload = result.as_dict()
    actions = payload["actions"]
    variant_action = next(a for a in actions if a["action"] == "register_variant")
    assert variant_action["metadata"]["scenario"] == "losing-streak"
    degrade_action = next(a for a in actions if a["action"] == "degrade_base")
    assert degrade_action["tactic_id"] == "momentum_core"


def test_evolution_manager_applies_parameter_mutation() -> None:
    router = PolicyRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="mean_rev_core",
            base_weight=1.0,
            parameters={"lookback": 20.0, "zscore_entry": 1.2},
            description="Baseline mean reversion",
        )
    )

    manager = EvolutionManager(
        policy_router=router,
        strategies=(
            ManagedStrategyConfig(
                base_tactic_id="mean_rev_core",
                degrade_multiplier=0.5,
                parameter_mutations=(
                    ParameterMutation(
                        parameter="lookback",
                        scale=1.2,
                        suffix="lookback_up",
                        rationale="Increase lookback after losses",
                        weight_multiplier=0.75,
                    ),
                ),
            ),
        ),
        window_size=2,
        win_rate_threshold=0.5,
        feature_flags=EvolutionFeatureFlags(env={"EVOLUTION_ENABLE_ADAPTIVE_RUNS": "1"}),
    )

    decision = _decision("mean_rev_core")
    result = None
    for _ in range(2):
        result = manager.observe_iteration(
            decision=decision,
            stage=PolicyLedgerStage.PAPER,
            outcomes={"paper_return": -0.02},
        )

    assert isinstance(result, EvolutionAdaptationResult)
    tactics = router.tactics()
    assert tactics["mean_rev_core"].base_weight == pytest.approx(0.5)

    mutated_id = "mean_rev_core__mut_lookback_up_1"
    assert mutated_id in tactics
    mutated = tactics[mutated_id]
    assert mutated.parameters["lookback"] == pytest.approx(24.0)
    assert mutated.base_weight == pytest.approx(0.75)

    payload = result.as_dict()
    actions = payload["actions"]
    assert any(action["action"] == "register_variant" and action["tactic_id"] == mutated_id for action in actions)
