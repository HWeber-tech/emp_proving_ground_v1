from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    timestamp: datetime | None = None,
) -> RegimeState:
    return RegimeState(
        regime=regime,
        confidence=confidence,
        features={
            "volume_z": volume_z,
            "volatility": volatility,
        },
        timestamp=timestamp or datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc),
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


def test_update_tactic_replaces_existing_definition() -> None:
    router = PolicyRouter()
    original = PolicyTactic(tactic_id="alpha", base_weight=1.0, description="old")
    router.register_tactic(original)

    updated = PolicyTactic(
        tactic_id="alpha",
        base_weight=1.2,
        description="new",
        objectives=("alpha-capture",),
        tags=("momentum",),
    )
    router.update_tactic(updated)

    stored = router.tactics()["alpha"]
    assert stored.description == "new"
    assert stored.base_weight == pytest.approx(1.2)
    assert stored.objectives == ("alpha-capture",)
    assert stored.tags == ("momentum",)

    with pytest.raises(KeyError):
        router.update_tactic(PolicyTactic(tactic_id="missing", base_weight=0.5))


def test_register_experiments_bulk() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="alpha", base_weight=1.0))

    experiments = (
        FastWeightExperiment(
            experiment_id="exp-one",
            tactic_id="alpha",
            delta=0.2,
            rationale="Boost alpha",
        ),
        FastWeightExperiment(
            experiment_id="exp-two",
            tactic_id="alpha",
            delta=-0.1,
            rationale="Throttle alpha",
        ),
    )

    router.register_experiments(experiments)

    assert set(router.experiments()) == {"exp-one", "exp-two"}


def test_reflection_digest_surfaces_emerging_strategies() -> None:
    router = PolicyRouter(reflection_history=10)
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="breakout",
                base_weight=1.0,
                regime_bias={"bull": 1.4},
                description="Momentum breakout",
                objectives=("alpha-capture", "trend-follow"),
                tags=("momentum", "fast-weight"),
            ),
            PolicyTactic(
                tactic_id="mean_reversion",
                base_weight=0.9,
                regime_bias={"bull": 0.9},
                description="Calm reversion",
                objectives=("stability",),
                tags=("reversion",),
            ),
        )
    )

    base = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    router.route(_regime(timestamp=base))
    router.route(_regime(timestamp=base + timedelta(minutes=5)))

    router.register_experiment(
        FastWeightExperiment(
            experiment_id="exp-reversion",
            tactic_id="mean_reversion",
            delta=0.75,
            rationale="Boost reversion while volatility remains muted",
            min_confidence=0.6,
            feature_gates={"volatility": (None, 0.3)},
        )
    )
    router.route(_regime(volatility=0.25, timestamp=base + timedelta(minutes=10)))

    digest = router.reflection_digest()

    assert digest["total_decisions"] == 3
    assert digest["as_of"].endswith("+00:00")

    top_tactic = digest["tactics"][0]
    assert top_tactic["tactic_id"] == "breakout"
    assert top_tactic["count"] == 2
    assert top_tactic["share"] == pytest.approx(2 / 3)
    assert set(top_tactic["tags"]) == {"momentum", "fast-weight"}

    reversion_entry = next(item for item in digest["tactics"] if item["tactic_id"] == "mean_reversion")
    assert reversion_entry["avg_score"] > 0.0
    assert reversion_entry["objectives"] == ["stability"]

    experiments = digest["experiments"]
    assert experiments[0]["experiment_id"] == "exp-reversion"
    assert experiments[0]["count"] == 1
    assert experiments[0]["most_common_tactic"] == "mean_reversion"

    tag_entries = digest["tags"]
    assert [entry["tag"] for entry in tag_entries[:2]] == ["fast-weight", "momentum"]
    assert tag_entries[0]["count"] == 2
    assert tag_entries[0]["top_tactics"][0] == "breakout"

    objective_entries = digest["objectives"]
    assert objective_entries[0]["objective"] == "alpha-capture"
    assert objective_entries[0]["share"] == pytest.approx(2 / 3)
    assert "breakout" in objective_entries[0]["top_tactics"]

    regimes = digest["regimes"]
    assert regimes["bull"]["share"] == pytest.approx(1.0)

    assert digest["current_streak"] == {"tactic_id": "mean_reversion", "length": 1}
    assert digest["longest_streak"] == {"tactic_id": "breakout", "length": 2}

    headlines = digest["recent_headlines"]
    assert len(headlines) == 3
    assert headlines[-1].startswith("Selected mean_reversion")
