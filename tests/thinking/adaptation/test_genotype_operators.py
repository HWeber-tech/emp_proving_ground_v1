from __future__ import annotations

from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.thinking.adaptation import (
    GenotypeOperatorResult,
    StrategyExecutionTopology,
    StrategyFeature,
    StrategyGenotype,
    StrategyRiskTemplate,
    op_add_feature,
    op_drop_feature,
    op_swap_execution_topology,
    op_tighten_risk,
)


def _baseline_genotype() -> StrategyGenotype:
    return StrategyGenotype(
        strategy_id="momentum_v1",
        features=(
            StrategyFeature(
                name="trend_strength",
                inputs=("price", "volume"),
                parameters={"window": 20, "threshold": 0.5},
            ),
            StrategyFeature(
                name="volatility_floor",
                inputs=("price",),
                parameters={"window": 10, "floor": 0.02},
            ),
        ),
        execution_topology=StrategyExecutionTopology(
            name="single_leg",
            version="v1",
            parameters={"latency_budget_ms": 50},
        ),
        risk_template=StrategyRiskTemplate(
            template_id="baseline",
            base_config=RiskConfig(),
            overrides={"max_risk_per_trade_pct": Decimal("0.02")},
        ),
        metadata={"family": "momentum"},
    )


def test_op_add_feature_appends_new_feature() -> None:
    genotype = _baseline_genotype()
    new_feature = StrategyFeature(
        name="order_flow_bias",
        inputs=("depth",),
        parameters={"window": 5},
    )

    result = op_add_feature(genotype, new_feature)

    assert isinstance(result, GenotypeOperatorResult)
    assert result.action == "op_add_feature"
    assert len(result.genotype.features) == 3
    assert result.genotype.features[-1].name == "order_flow_bias"
    assert result.metadata["feature"] == "order_flow_bias"
    assert result.metadata["position"] == 2


def test_op_add_feature_replaces_existing_when_requested() -> None:
    genotype = _baseline_genotype()
    replacement = StrategyFeature(
        name="trend_strength",
        inputs=("price",),
        parameters={"window": 30, "threshold": 0.6},
    )

    result = op_add_feature(genotype, replacement, replace=True)

    names = [feature.name for feature in result.genotype.features]
    assert names.count("trend_strength") == 1
    feature = next(feature for feature in result.genotype.features if feature.name == "trend_strength")
    assert feature.parameters["window"] == 30
    assert result.metadata["replaced_feature"] == "trend_strength"


def test_op_add_feature_raises_on_duplicate_without_replace() -> None:
    genotype = _baseline_genotype()

    with pytest.raises(ValueError):
        op_add_feature(
            genotype,
            {
                "name": "trend_strength",
                "inputs": ("price",),
                "parameters": {"window": 40},
            },
        )


def test_op_drop_feature_removes_feature() -> None:
    genotype = _baseline_genotype()

    result = op_drop_feature(genotype, "volatility_floor")

    remaining = [feature.name for feature in result.genotype.features]
    assert remaining == ["trend_strength"]
    assert result.metadata["feature"] == "volatility_floor"
    assert result.metadata["dropped"] is True


def test_op_drop_feature_missing_ok_returns_original() -> None:
    genotype = _baseline_genotype()

    result = op_drop_feature(genotype, "unknown_feature", missing_ok=True)

    assert result.genotype is genotype
    assert result.metadata["dropped"] is False


def test_op_swap_execution_topology_replaces_topology() -> None:
    genotype = _baseline_genotype()

    result = op_swap_execution_topology(
        genotype,
        {
            "name": "dual_leg",
            "version": "v2",
            "parameters": {"latency_budget_ms": 80, "path": "alt"},
        },
    )

    assert result.genotype.execution_topology.name == "dual_leg"
    assert result.metadata["previous_topology"] == "single_leg"
    assert result.metadata["new_topology"] == "dual_leg"


def test_op_tighten_risk_scales_percentages_and_applies_floors() -> None:
    genotype = _baseline_genotype()

    result = op_tighten_risk(
        genotype,
        scale=0.5,
        floors={"max_risk_per_trade_pct": Decimal("0.015")},
    )

    overrides = result.genotype.risk_template.overrides
    assert overrides["max_risk_per_trade_pct"] == Decimal("0.0150")
    assert overrides["max_total_exposure_pct"] == Decimal("0.2500")
    fields = {entry["field"] for entry in result.metadata["updated_fields"]}
    assert {"max_risk_per_trade_pct", "max_total_exposure_pct"}.issubset(fields)


def test_genotype_operator_result_as_dict_serialises() -> None:
    genotype = _baseline_genotype()
    result = op_add_feature(genotype, StrategyFeature(name="imbalance", inputs=("depth",), parameters={}))
    payload = result.as_dict()

    assert payload["action"] == "op_add_feature"
    assert payload["genotype"]["strategy_id"] == "momentum_v1"
