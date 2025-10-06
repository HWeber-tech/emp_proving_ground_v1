from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.thinking.adaptation.policy_router import (
    FastWeightExperiment,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)


def _regime(
    regime: str = "bull",
    confidence: float = 0.8,
    *,
    volume_z: float = 0.1,
    volatility: float = 0.2,
) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        features={
            "volume_z": volume_z,
            "volatility": volatility,
        },
        timestamp=datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc),
    )


def test_route_selects_highest_weight_with_regime_bias() -> None:
    router = PolicyRouter(default_guardrails={"max_latency_ms": 250})
    router.register_tactic(
        PolicyTactic(
            tactic_id="breakout",
            base_weight=1.0,
            parameters={"style": "momentum"},
            guardrails={"requires_diary": True},
            regime_bias={"bull": 1.4},
            confidence_sensitivity=0.8,
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="mean_reversion",
            base_weight=1.1,
            parameters={"style": "revert"},
            guardrails={"requires_diary": True},
            regime_bias={"bear": 1.5},
        )
    )

    decision = router.route(_regime())

    assert decision.tactic_id == "breakout"
    assert decision.guardrails["max_latency_ms"] == 250
    assert decision.parameters["regime_hint"] == "bull"
    assert decision.reflection_summary["headline"].startswith("Selected breakout")


def test_fast_weight_experiment_overrides_base_score() -> None:
    router = PolicyRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="breakout",
            base_weight=1.0,
            regime_bias={"bull": 1.4},
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="mean_reversion",
            base_weight=0.9,
            regime_bias={"bull": 0.9},
        )
    )
    router.register_experiment(
        FastWeightExperiment(
            experiment_id="exp-fast-weights",
            tactic_id="mean_reversion",
            delta=0.8,
            rationale="Promote reversion while volatility is muted",
            min_confidence=0.6,
            feature_gates={"volatility": (None, 0.3)},
        )
    )

    decision = router.route(_regime(volatility=0.25))

    assert decision.tactic_id == "mean_reversion"
    assert decision.experiments_applied == ("exp-fast-weights",)
    summary_experiments = decision.reflection_summary["experiments"]
    assert summary_experiments[0]["experiment_id"] == "exp-fast-weights"
    assert "Promote reversion" in summary_experiments[0]["rationale"]


def test_route_respects_external_fast_weights() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="base", base_weight=1.0))
    router.register_tactic(PolicyTactic(tactic_id="alt", base_weight=0.7))

    decision = router.route(_regime(), fast_weights={"alt": 2.0})

    assert decision.tactic_id == "alt"
    top_candidates = decision.reflection_summary["top_candidates"]
    assert {candidate["tactic_id"] for candidate in top_candidates} == {"base", "alt"}


def test_registering_duplicate_tactic_raises() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="dup", base_weight=1.0))

    with pytest.raises(ValueError):
        router.register_tactic(PolicyTactic(tactic_id="dup", base_weight=1.0))
