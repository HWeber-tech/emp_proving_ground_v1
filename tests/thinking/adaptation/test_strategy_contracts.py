from __future__ import annotations

from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.thinking.adaptation.strategy_contracts import (
    StrategyExecutionTopology,
    StrategyFeature,
    StrategyGenotype,
    StrategyRiskTemplate,
    StrategyTunable,
)


def _baseline_genotype() -> StrategyGenotype:
    base_risk = RiskConfig()
    return StrategyGenotype(
        strategy_id="trend_alpha",
        features=(
            StrategyFeature(
                name="trend_strength",
                inputs=("price", "volume"),
                parameters={"window": 20, "threshold": 0.5},
                description="Primary momentum signal",
                economic_hypothesis=(
                    "If realised momentum with volume confirmation exceeds the threshold, "
                    "trend continuation should deliver positive excess returns; falsified when "
                    "forward returns turn negative under the same condition."
                ),
                ci_tests=(
                    "tests/thinking/adaptation/test_strategy_contracts.py::"
                    "test_strategy_genotype_realise_builds_phenotype_with_overrides",
                ),
            ),
            StrategyFeature(
                name="volatility_floor",
                inputs=("price",),
                parameters={"window": 10, "floor": 0.01},
                economic_hypothesis=(
                    "If trailing volatility falls below the configured floor, position sizing "
                    "must contract to avoid drawdowns exceeding the risk budget; falsified when "
                    "drawdowns remain controlled without the floor."
                ),
                ci_tests=(
                    "tests/thinking/adaptation/test_strategy_contracts.py::"
                    "test_strategy_genotype_realise_builds_phenotype_with_overrides",
                ),
            ),
        ),
        execution_topology=StrategyExecutionTopology(
            name="single_leg",
            version="v1",
            parameters={"latency_budget_ms": 50},
        ),
        risk_template=StrategyRiskTemplate(
            template_id="baseline",
            base_config=base_risk,
            overrides={"max_risk_per_trade_pct": Decimal("0.015")},
        ),
        tunables=(
            StrategyTunable(
                name="lookback",
                default=30,
                minimum=5,
                maximum=120,
                description="Lookback window for primary momentum feature",
            ),
            StrategyTunable(
                name="risk_multiplier",
                default=1.0,
                minimum=0.5,
                maximum=2.0,
            ),
        ),
        metadata={"family": "momentum"},
    )


def test_strategy_genotype_realise_builds_phenotype_with_overrides() -> None:
    genotype = _baseline_genotype()

    phenotype = genotype.realise(
        tunable_overrides={"lookback": 45},
        risk_overrides={"max_total_exposure_pct": Decimal("0.40")},
        feature_parameter_overrides={"trend_strength": {"window": 25}},
        topology_overrides={"latency_budget_ms": 35, "path": "slow-lane"},
        metadata={"note": "paper trial"},
    )

    assert phenotype.strategy_id == "trend_alpha"
    trend_feature = next(feature for feature in phenotype.features if feature.name == "trend_strength")
    assert trend_feature.parameters["window"] == 25
    assert trend_feature.parameters["threshold"] == 0.5

    topology = phenotype.execution_topology
    assert topology.parameters["latency_budget_ms"] == 35
    assert topology.parameters["path"] == "slow-lane"

    assert phenotype.tunable_values["lookback"] == 45
    assert phenotype.tunable_values["risk_multiplier"] == 1.0

    assert phenotype.metadata["family"] == "momentum"
    assert phenotype.metadata["note"] == "paper trial"

    assert phenotype.risk_config.max_total_exposure_pct == Decimal("0.40")
    assert phenotype.risk_config.max_risk_per_trade_pct == Decimal("0.015")


def test_strategy_genotype_rejects_duplicate_feature_names() -> None:
    feature = StrategyFeature(
        name="alpha",
        inputs=("price",),
        parameters={},
        economic_hypothesis=(
            "If the alpha feature activates on price dislocations, trade expectancy increases; "
            "falsified when trades under activation underperform baseline."
        ),
        ci_tests=(
            "tests/thinking/adaptation/test_strategy_contracts.py::"
            "test_strategy_genotype_rejects_duplicate_feature_names",
        ),
    )
    risk_template = StrategyRiskTemplate(template_id="base")
    topology = StrategyExecutionTopology(name="direct")

    with pytest.raises(ValueError):
        StrategyGenotype(
            strategy_id="dup",
            features=(feature, feature),
            execution_topology=topology,
            risk_template=risk_template,
        )


def test_tunable_enforces_bounds_and_unknown_overrides_raise() -> None:
    genotype = _baseline_genotype()

    assert genotype.resolve_tunables({"lookback": 100})["lookback"] == 100

    with pytest.raises(ValueError):
        genotype.resolve_tunables({"lookback": 200})

    with pytest.raises(TypeError):
        genotype.resolve_tunables({"lookback": "not numeric"})

    with pytest.raises(KeyError):
        genotype.resolve_tunables({"unknown": 1})


def test_strategy_feature_requires_hypothesis_and_ci_tests() -> None:
    with pytest.raises(ValueError):
        StrategyFeature(
            name="incomplete",
            inputs=("price",),
            parameters={},
            economic_hypothesis="",
            ci_tests=(
                "tests/thinking/adaptation/test_strategy_contracts.py::"
                "test_strategy_feature_requires_hypothesis_and_ci_tests",
            ),
        )

    with pytest.raises(ValueError):
        StrategyFeature(
            name="missing_tests",
            inputs=("price",),
            parameters={},
            economic_hypothesis=(
                "If liquidity premium widens, the strategy captures positive drift; falsified "
                "when returns stay neutral despite the signal."
            ),
            ci_tests=(),
        )
