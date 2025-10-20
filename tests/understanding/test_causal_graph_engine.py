from __future__ import annotations

import math

import pytest

from src.understanding.causal_graph_engine import CausalGraphEngine


def _build_context() -> dict[str, float]:
    return {
        "macro_signal": 0.2,
        "base_liquidity": 1.5,
        "liquidity_shock": 0.1,
        "macro_to_liquidity_beta": 0.5,
        "mid_price": 100.0,
        "order_imbalance": 0.4,
        "microprice_sensitivity": 0.75,
        "spread": 0.05,
        "limit_price": 100.12,
        "fill_urgency": 0.3,
        "order_size": 2.0,
    }


def test_default_graph_structure_links_macro_to_fills() -> None:
    engine = CausalGraphEngine.default()

    assert engine.topology == ("macro", "liquidity", "microprice", "fills")
    assert set(engine.edges) == {
        ("macro", "liquidity"),
        ("liquidity", "microprice"),
        ("microprice", "fills"),
    }


def test_evaluate_returns_expected_values() -> None:
    engine = CausalGraphEngine.default()
    context = _build_context()

    result = engine.evaluate(context)

    expected_macro = context["macro_signal"]
    expected_liquidity = max(
        context["base_liquidity"]
        * (1.0 + context["macro_to_liquidity_beta"] * expected_macro)
        - context["liquidity_shock"],
        0.0,
    )
    liquidity_effect = 1.0 / (1.0 + expected_liquidity)
    expected_micro = context["mid_price"] + (
        context["order_imbalance"]
        * context["spread"]
        * context["microprice_sensitivity"]
        * liquidity_effect
    )
    edge = (context["limit_price"] - expected_micro) / context["spread"] + context["fill_urgency"]
    expected_fill = context["order_size"] / (1.0 + math.exp(-edge))

    assert result["macro"] == pytest.approx(expected_macro)
    assert result["liquidity"] == pytest.approx(expected_liquidity)
    assert result["microprice"] == pytest.approx(expected_micro)
    assert result["fills"] == pytest.approx(expected_fill)


def test_macro_intervention_propagates_through_graph() -> None:
    engine = CausalGraphEngine.default()
    context = _build_context()

    intervention_result = engine.run_intervention(context, {"macro": 0.8})

    assert intervention_result.baseline["macro"] == pytest.approx(0.2)
    assert intervention_result.intervened["macro"] == pytest.approx(0.8)
    assert intervention_result.delta["liquidity"] > 0.0
    assert intervention_result.delta["microprice"] < 0.0
    assert intervention_result.delta["fills"] > 0.0
    assert intervention_result.affected_nodes == (
        "macro",
        "liquidity",
        "microprice",
        "fills",
    )
