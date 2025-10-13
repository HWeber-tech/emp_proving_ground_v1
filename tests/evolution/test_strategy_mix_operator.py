from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.evolution.mutation import (
    StrategyMixCandidate,
    StrategyMixState,
    op_mix_strategies,
)


def assert_weights_sum_to_one(weights: dict[str, float]) -> None:
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_mix_prioritises_scores() -> None:
    candidates = [
        StrategyMixCandidate(tactic_id="alpha", score=2.0),
        StrategyMixCandidate(tactic_id="beta", score=1.0),
    ]

    result = op_mix_strategies(
        candidates,
        friction=0.0,
        max_components=2,
        min_share=0.05,
        max_share=0.8,
    )

    weights = dict(result.weights)
    assert set(weights) == {"alpha", "beta"}
    assert_weights_sum_to_one(weights)
    assert weights["alpha"] > weights["beta"]
    assert result.penalty == pytest.approx(0.0, abs=1e-9)


def test_switching_friction_slows_reallocation() -> None:
    previous_state = StrategyMixState(
        weights={"alpha": 0.75, "beta": 0.25},
        timestamp=datetime(2025, 1, 1, 12, tzinfo=UTC),
    )
    candidates = [
        StrategyMixCandidate(tactic_id="beta", score=0.9),
        StrategyMixCandidate(tactic_id="gamma", score=0.8),
    ]

    result = op_mix_strategies(
        candidates,
        previous_state=previous_state,
        friction=0.6,
        max_components=3,
        min_share=0.05,
        max_share=0.8,
        timestamp=datetime(2025, 1, 1, 12, 5, tzinfo=UTC),
    )

    weights = dict(result.weights)
    assert_weights_sum_to_one(weights)
    # Legacy tactic alpha should persist with a non-trivial share because of friction.
    assert weights["alpha"] > 0.3
    # Newcomer gamma should enter the mix but remain below its target share due to friction.
    assert "gamma" in weights
    assert weights["gamma"] < 0.35
    assert result.penalty > 0.0


def test_friction_decays_with_half_life() -> None:
    previous_state = StrategyMixState(
        weights={"alpha": 0.7, "beta": 0.3},
        timestamp=datetime(2025, 2, 1, 10, tzinfo=UTC),
    )
    candidates = [
        StrategyMixCandidate(tactic_id="beta", score=0.9),
        StrategyMixCandidate(tactic_id="gamma", score=0.8),
    ]

    late_timestamp = previous_state.timestamp + timedelta(hours=6)
    result = op_mix_strategies(
        candidates,
        previous_state=previous_state,
        friction=0.6,
        max_components=3,
        min_share=0.05,
        max_share=0.8,
        decay_half_life=3600.0,
        timestamp=late_timestamp,
    )

    weights = dict(result.weights)
    assert_weights_sum_to_one(weights)
    # Effective friction should almost vanish after a long gap.
    assert result.effective_friction < 0.05
    # Mix should be close to the target allocation (~0.53 / 0.47) when friction decays.
    assert weights.get("beta", 0.0) == pytest.approx(0.53, rel=0.1)
    assert weights.get("gamma", 0.0) == pytest.approx(0.47, rel=0.1)


def test_candidate_max_share_enforced() -> None:
    candidates = [
        StrategyMixCandidate(tactic_id="alpha", score=10.0, max_weight=0.5),
        StrategyMixCandidate(tactic_id="beta", score=1.0),
        StrategyMixCandidate(tactic_id="gamma", score=1.0),
    ]

    result = op_mix_strategies(
        candidates,
        friction=0.0,
        max_components=3,
        min_share=0.05,
        max_share=0.8,
    )

    weights = dict(result.weights)
    assert_weights_sum_to_one(weights)
    assert weights["alpha"] == pytest.approx(0.5, abs=1e-9)
