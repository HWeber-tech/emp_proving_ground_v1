from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.understanding.causal_graph_engine import CausalGraphEngine
from src.understanding.counterfactual_simulator import (
    CounterfactualAssessment,
    CounterfactualScenario,
    CounterfactualSimulator,
)
from src.understanding.router import BeliefSnapshot
from src.thinking.adaptation.policy_router import RegimeState


def _build_belief_snapshot() -> BeliefSnapshot:
    regime_state = RegimeState(
        regime="normal",
        confidence=0.8,
        features={},
        timestamp=datetime.now(timezone.utc),
    )
    features = {
        "macro_signal": 0.25,
        "base_liquidity": 1.4,
        "liquidity_shock": 0.05,
        "macro_to_liquidity_beta": 0.55,
        "mid_price": 100.0,
        "order_imbalance": 0.3,
        "microprice_sensitivity": 0.65,
        "spread": 0.05,
        "limit_price": 100.1,
        "fill_urgency": 0.15,
        "order_size": 1.5,
    }
    return BeliefSnapshot(
        belief_id="belief.test",
        regime_state=regime_state,
        features=features,
        metadata={"symbol": "TEST"},
    )


def test_counterfactual_simulator_runs_interventions() -> None:
    belief = _build_belief_snapshot()
    engine = CausalGraphEngine.default()
    simulator = CounterfactualSimulator(engine, tolerance=0.2)
    scenario = CounterfactualScenario(
        name="macro_nudge",
        interventions={"macro": 0.3},
        description="Small increase in macro signal",
    )

    assessment = simulator.simulate(belief, [scenario])

    assert isinstance(assessment, CounterfactualAssessment)
    context = {key: float(value) for key, value in belief.features.items()}
    expected_baseline = engine.evaluate(context)
    for key, value in expected_baseline.items():
        assert assessment.baseline[key] == pytest.approx(value)

    result = assessment.scenarios[0]
    expected_intervention = engine.run_intervention(context, scenario.interventions)
    for key, value in expected_intervention.intervened.items():
        assert result.intervention.intervened[key] == pytest.approx(value)
    assert result.robust is True
    assert assessment.robust is True


def test_counterfactual_simulator_marks_unstable_scenario() -> None:
    belief = _build_belief_snapshot()
    engine = CausalGraphEngine.default()
    simulator = CounterfactualSimulator(engine, tolerance=0.05)
    scenario = CounterfactualScenario(
        name="macro_shock",
        interventions={"macro": -1.2},
        description="Significant negative macro shock",
    )

    assessment = simulator.simulate(belief, [scenario])
    result = assessment.scenarios[0]

    assert result.max_abs_delta > 0.05
    assert result.robust is False
    assert assessment.robust is False


def test_counterfactual_simulator_respects_context_overrides() -> None:
    belief = _build_belief_snapshot()
    engine = CausalGraphEngine.default()
    simulator = CounterfactualSimulator(engine)
    overrides = {"spread": 0.02}
    scenario = CounterfactualScenario(name="baseline", interventions={})

    assessment = simulator.simulate(belief, [scenario], context_overrides=overrides)

    context = {key: float(value) for key, value in belief.features.items()}
    context.update({key: float(value) for key, value in overrides.items()})
    expected_baseline = engine.evaluate(context)

    for key, value in expected_baseline.items():
        assert assessment.baseline[key] == pytest.approx(value)
    assert assessment.scenarios[0].intervention.baseline["microprice"] == pytest.approx(
        expected_baseline["microprice"]
    )

