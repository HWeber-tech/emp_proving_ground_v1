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
    breakdown = decision.weight_breakdown
    assert breakdown["base_weight"] == pytest.approx(1.0)
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["experiment_multipliers"] == {}
    assert breakdown["final_score"] == pytest.approx(decision.selected_weight)


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
            regimes=("bull",),
        )
    )

    decision = router.route(_regime(volatility=0.25))

    assert decision.tactic_id == "mean_reversion"
    assert decision.experiments_applied == ("exp-fast-weights",)
    summary_experiments = decision.reflection_summary["experiments"]
    assert summary_experiments[0]["experiment_id"] == "exp-fast-weights"
    assert "Promote reversion" in summary_experiments[0]["rationale"]
    assert summary_experiments[0]["regimes"] == ["bull"]
    assert summary_experiments[0]["min_confidence"] == pytest.approx(0.6)
    assert summary_experiments[0]["delta"] == pytest.approx(0.8)
    gates = summary_experiments[0]["feature_gates"]
    assert isinstance(gates, list) and gates[0]["feature"] == "volatility"
    assert gates[0]["maximum"] == pytest.approx(0.3)
    breakdown = decision.weight_breakdown
    assert breakdown["experiment_multipliers"]["exp-fast-weights"] == pytest.approx(1.8)
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["total_multiplier"] == pytest.approx(1.8)
    assert breakdown["final_score"] == pytest.approx(decision.selected_weight)
    assert decision.reflection_summary["weight_breakdown"]["total_multiplier"] == pytest.approx(1.8)


def test_route_respects_external_fast_weights() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="base", base_weight=1.0))
    router.register_tactic(PolicyTactic(tactic_id="alt", base_weight=0.7))

    decision = router.route(_regime(), fast_weights={"alt": 2.0})

    assert decision.tactic_id == "alt"
    top_candidates = decision.reflection_summary["top_candidates"]
    assert {candidate["tactic_id"] for candidate in top_candidates} == {"base", "alt"}
    breakdown = decision.weight_breakdown
    assert breakdown["fast_weight_multiplier"] == pytest.approx(2.0)
    assert breakdown["total_multiplier"] == pytest.approx(2.0)
    assert breakdown["experiment_multipliers"] == {}


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
            regimes=("bull",),
        )
    )
    router.route(_regime(volatility=0.25, timestamp=base + timedelta(minutes=10)))

    digest = router.reflection_digest()

    assert digest["total_decisions"] == 3
    assert digest["as_of"].endswith("+00:00")

    confidence = digest["confidence"]
    assert confidence["count"] == 3
    assert confidence["average"] == pytest.approx(0.8)
    assert confidence["latest"] == pytest.approx(0.8)
    assert confidence["change"] == pytest.approx(0.0)
    assert confidence["first_seen"] is not None
    assert confidence["last_seen"].endswith("+00:00")

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
    assert experiments[0]["regimes"] == ["bull"]
    assert experiments[0]["min_confidence"] == pytest.approx(0.6)
    assert experiments[0]["multiplier"] == pytest.approx(1.75)
    assert experiments[0]["delta"] == pytest.approx(0.75)
    gates = experiments[0]["feature_gates"]
    assert isinstance(gates, list) and gates[0]["maximum"] == pytest.approx(0.3)

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

    feature_entries = digest["features"]
    volatility_entry = next(item for item in feature_entries if item["feature"] == "volatility")
    assert volatility_entry["count"] == 3
    assert volatility_entry["latest"] == pytest.approx(0.25)
    assert volatility_entry["trend"] == pytest.approx(0.05)

    assert digest["current_streak"] == {"tactic_id": "mean_reversion", "length": 1}
    assert digest["longest_streak"] == {"tactic_id": "breakout", "length": 2}

    headlines = digest["recent_headlines"]
    assert len(headlines) == 3
    assert headlines[-1].startswith("Selected mean_reversion")
    weight_stats = digest["weight_stats"]
    assert weight_stats["fast_weight"]["applications"] == 0
    assert weight_stats["fast_weight"]["average_multiplier"] == pytest.approx(1.0)
    assert weight_stats["average_total_multiplier"] == pytest.approx((1.0 + 1.0 + 1.75) / 3)
    assert weight_stats["average_final_score"] == pytest.approx(
        sum(entry["score"] for entry in router.history()) / 3
    )


def test_prune_experiments_removes_expired_entries() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="alpha", base_weight=1.0))

    expired = FastWeightExperiment(
        experiment_id="exp-expired",
        tactic_id="alpha",
        delta=0.2,
        rationale="Promote alpha temporarily",
        expires_at=datetime(2024, 3, 15, 12, 0),
    )
    active = FastWeightExperiment(
        experiment_id="exp-active",
        tactic_id="alpha",
        delta=0.1,
        rationale="Ongoing alpha boost",
        expires_at=datetime(2024, 3, 15, 13, 0, tzinfo=timezone.utc),
    )
    router.register_experiment(expired)
    router.register_experiment(active)

    removed = router.prune_experiments(now=datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc))

    assert set(removed) == {"exp-expired"}
    assert "exp-expired" not in router.experiments()
    assert "exp-active" in router.experiments()


def test_reflection_report_wraps_builder_helpers() -> None:
    base = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    router = PolicyRouter()
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="breakout",
                base_weight=1.0,
                regime_bias={"bull": 1.3},
                description="Momentum breakout",
                objectives=("alpha",),
                tags=("momentum",),
            ),
            PolicyTactic(
                tactic_id="mean_reversion",
                base_weight=0.9,
                regime_bias={"bull": 1.0},
                description="Reversion",
                objectives=("stability",),
                tags=("reversion",),
            ),
        )
    )
    router.route(_regime(timestamp=base))
    router.route(_regime(timestamp=base + timedelta(minutes=3)))
    router.route(_regime(timestamp=base + timedelta(minutes=6)))

    generated_at = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)
    artifacts = router.reflection_report(now=generated_at, max_tactics=2, max_experiments=2)

    assert artifacts.payload["metadata"]["generated_at"] == generated_at.isoformat()
    assert artifacts.digest["total_decisions"] == 3
    assert artifacts.digest["confidence"]["count"] == 3
    assert "PolicyRouter reflection summary" in artifacts.markdown


def test_ingest_reflection_history_replays_summaries() -> None:
    router = PolicyRouter(reflection_history=10)
    base = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

    entries = (
        {
            "headline": "Selected alpha for bull",
            "tactic_id": "alpha",
            "score": 1.25,
            "regime": "bull",
            "timestamp": base.isoformat(),
            "tactic_tags": ["momentum"],
            "tactic_objectives": ["alpha"],
        },
        {
            "headline": "Selected beta for bear",
            "tactic_id": "beta",
            "score": 0.95,
            "regime": "bear",
            "timestamp": (base + timedelta(minutes=5)),
            "tactic_tags": ("reversion",),
            "tactic_objectives": ("stability",),
            "experiments": (
                {
                    "experiment_id": "exp-rebalance",
                    "tactic_id": "beta",
                    "multiplier": 1.1,
                    "rationale": "Promote bear defence",
                },
            ),
        },
    )

    appended = router.ingest_reflection_history(entries)

    assert appended == 2
    history = router.history()
    assert len(history) == 2
    assert history[0]["tactic_id"] == "alpha"
    assert history[1]["tactic_id"] == "beta"

    digest = router.reflection_digest()
    assert digest["total_decisions"] == 2
    assert set(entry["tactic_id"] for entry in digest["tactics"]) == {"alpha", "beta"}
    experiments = digest["experiments"]
    assert experiments[0]["experiment_id"] == "exp-rebalance"


def test_ingest_reflection_history_skips_invalid_entries() -> None:
    router = PolicyRouter()

    entries = (
        {"headline": "missing timestamp", "tactic_id": "alpha"},
        {"headline": "bad tactic", "timestamp": "2024-03-15T12:00:00+00:00", "tactic_id": ""},
        {"headline": "invalid timestamp", "tactic_id": "beta", "timestamp": "not-iso"},
    )

    appended = router.ingest_reflection_history(entries)

    assert appended == 0
    assert router.history() == ()


def test_fast_weight_experiment_respects_regime_filters() -> None:
    router = PolicyRouter()
    router.register_tactics(
        (
            PolicyTactic(tactic_id="baseline", base_weight=1.0, regime_bias={"bull": 1.0}),
            PolicyTactic(tactic_id="bear_boost", base_weight=0.6, regime_bias={"bear": 1.2}),
        )
    )
    router.register_experiment(
        FastWeightExperiment(
            experiment_id="bear-only",
            tactic_id="bear_boost",
            delta=0.8,
            rationale="Promote bear defence",
            regimes=("bear",),
        )
    )

    bull_decision = router.route(_regime(regime="bull"))
    assert bull_decision.tactic_id == "baseline"
    assert bull_decision.experiments_applied == ()

    bear_decision = router.route(_regime(regime="bear"))
    assert bear_decision.tactic_id == "bear_boost"
    assert bear_decision.experiments_applied == ("bear-only",)


def test_experiment_registry_surfaces_metadata() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="alpha", base_weight=1.0))
    expires_at = datetime(2024, 3, 15, 13, 0)
    router.register_experiment(
        FastWeightExperiment(
            experiment_id="exp-alpha",
            tactic_id="alpha",
            delta=0.25,
            rationale="Boost alpha in bull regimes",
            min_confidence=0.55,
            feature_gates={"volatility": (None, 0.4)},
            expires_at=expires_at,
            regimes=("bull", "neutral"),
        )
    )

    snapshot = RegimeState(
        regime="bull",
        confidence=0.6,
        features={"volatility": 0.3},
        timestamp=datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc),
    )
    registry = router.experiment_registry(regime_state=snapshot)
    assert registry[0]["experiment_id"] == "exp-alpha"
    assert registry[0]["would_apply"] is True
    assert registry[0]["regimes"] == ["bull", "neutral"]
    assert registry[0]["feature_gates"][0]["maximum"] == pytest.approx(0.4)
    assert registry[0]["min_confidence"] == pytest.approx(0.55)
    assert registry[0]["expires_at"].startswith("2024-03-15T13:00:00")

    cold_snapshot = RegimeState(
        regime="bear",
        confidence=0.6,
        features={"volatility": 0.3},
        timestamp=snapshot.timestamp,
    )
    registry_cold = router.experiment_registry(regime_state=cold_snapshot)
    assert registry_cold[0]["would_apply"] is False


def test_tactic_catalog_surfaces_metadata() -> None:
    router = PolicyRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha",
            base_weight=1.1,
            parameters={"style": "momentum"},
            guardrails={"requires_diary": True},
            regime_bias={"bull": 1.3},
            confidence_sensitivity=0.7,
            description="Momentum alpha tactic",
            objectives=("alpha-capture",),
            tags=("momentum", "fast-weight"),
        )
    )

    catalogue = router.tactic_catalog()
    assert catalogue[0]["tactic_id"] == "alpha"
    assert catalogue[0]["guardrails"] == {"requires_diary": True}
    assert catalogue[0]["parameters"]["style"] == "momentum"
    assert catalogue[0]["objectives"] == ["alpha-capture"]
    assert "momentum" in catalogue[0]["tags"]
