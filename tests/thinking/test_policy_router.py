from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

import pytest

from src.thinking.adaptation.fast_weights import FastWeightConstraints, FastWeightController
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
    metrics = decision.fast_weight_metrics
    assert metrics["active_percentage"] == pytest.approx(0.0)
    assert metrics["total"] == 2


def test_regime_flip_forces_topology_switch_and_records_transition() -> None:
    router = PolicyRouter(regime_switch_deadline_ms=50)
    base = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="trend_hunter",
                base_weight=1.5,
                regime_bias={"bull": 1.4, "bear": 0.9},
                topology="topo-trend",
                description="Momentum breakout topology",
            ),
            PolicyTactic(
                tactic_id="defensive_wall",
                base_weight=0.6,
                regime_bias={"bear": 1.5},
                topology="topo-defensive",
                description="Defensive hedging topology",
            ),
        )
    )

    bull_regime = _regime(regime="bull", timestamp=base)
    bull_decision = router.route(bull_regime, decision_timestamp=base)

    assert bull_decision.tactic_id == "trend_hunter"
    assert bull_decision.parameters["execution_topology"] == "topo-trend"
    assert bull_decision.decision_timestamp is not None
    assert bull_decision.decision_timestamp.tzinfo is timezone.utc

    bear_regime = _regime(regime="bear", timestamp=base + timedelta(milliseconds=5))
    decision_time = bear_regime.timestamp + timedelta(milliseconds=30)
    bear_decision = router.route(bear_regime, decision_timestamp=decision_time)

    assert bear_decision.tactic_id == "defensive_wall"
    assert bear_decision.parameters["execution_topology"] == "topo-defensive"

    transition = bear_decision.reflection_summary["regime_transition"]
    assert transition["regime_changed"] is True
    assert transition["previous_tactic"] == "trend_hunter"
    assert transition["current_tactic"] == "defensive_wall"
    assert transition["topology_changed"] is True
    assert transition["switch_forced"] is True
    assert transition["met_deadline"] is True
    assert transition["latency_ms"] == pytest.approx(30.0, rel=1e-6, abs=1e-3)
    assert transition["current_topology"] == "topo-defensive"
    assert transition["previous_topology"] == "topo-trend"
    candidates = transition["topology_candidates"]
    assert isinstance(candidates, list) and candidates[0]["tactic_id"] == "defensive_wall"


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
    metrics = decision.fast_weight_metrics
    assert metrics["active_percentage"] == pytest.approx(0.0)


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
    assert breakdown["fast_weight_active_percentage"] == pytest.approx(50.0)
    metrics = decision.fast_weight_metrics
    assert metrics["active_percentage"] == pytest.approx(50.0)
    assert metrics["total"] == 2
    assert metrics["active"] == 1
    metrics = decision.reflection_summary["fast_weight_metrics"]
    assert metrics["active_percentage"] == pytest.approx(50.0)
    assert metrics["total"] == 2
    assert metrics["sparsity"] == pytest.approx(0.5)
    assert metrics["active_ids"] == ("alt",)
    assert metrics["dormant_ids"] == ("base",)


def test_route_clamps_negative_fast_weight_overrides() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="base", base_weight=1.0))
    router.register_tactic(PolicyTactic(tactic_id="alt", base_weight=0.8))

    decision = router.route(_regime(), fast_weights={"alt": -3.5})

    assert decision.tactic_id == "base"
    breakdown = decision.weight_breakdown
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["fast_weight_active_percentage"] == pytest.approx(0.0)
    metrics = decision.fast_weight_metrics
    assert metrics["active"] == 0
    assert metrics["min_multiplier"] == pytest.approx(0.0)
    assert metrics["max_multiplier"] == pytest.approx(1.0)
    assert all(value >= 0.0 for value in metrics.values() if isinstance(value, (int, float)))


def test_route_suppresses_inhibitory_when_constraints_disallow() -> None:
    controller = FastWeightController(FastWeightConstraints(allow_inhibitory=False))
    router = PolicyRouter(fast_weight_controller=controller)
    router.register_tactic(PolicyTactic(tactic_id="base", base_weight=1.0))
    router.register_tactic(PolicyTactic(tactic_id="alt", base_weight=0.8))

    decision = router.route(_regime(), fast_weights={"alt": 0.6})

    assert decision.tactic_id == "base"
    breakdown = decision.weight_breakdown
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["fast_weight_active_percentage"] == pytest.approx(0.0)
    metrics = decision.fast_weight_metrics
    assert metrics["inhibitory"] == 0
    assert metrics["suppressed_inhibitory"] == 1
    assert metrics["suppressed_inhibitory_ids"] == ("alt",)


def test_fast_weight_metrics_zero_when_no_adjustments() -> None:
    router = PolicyRouter()
    router.register_tactic(PolicyTactic(tactic_id="solo", base_weight=1.0))

    decision = router.route(_regime())

    breakdown = decision.weight_breakdown
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["fast_weight_active_percentage"] == pytest.approx(0.0)
    metrics = decision.fast_weight_metrics
    assert metrics["active_percentage"] == pytest.approx(0.0)
    metrics = decision.reflection_summary["fast_weight_metrics"]
    assert metrics["active_percentage"] == pytest.approx(0.0)
    assert metrics["active"] == 0
    assert metrics["total"] == 1


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
    assert top_tactic["first_seen"].endswith("+00:00")
    assert top_tactic["last_seen"].endswith("+00:00")

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
    assert experiments[0]["first_seen"].endswith("+00:00")
    assert experiments[0]["last_seen"].endswith("+00:00")

    emerging_tactics = digest["emerging_tactics"]
    assert emerging_tactics[0]["tactic_id"] == "mean_reversion"
    assert emerging_tactics[0]["count"] == 1
    assert emerging_tactics[0]["first_seen"].endswith("+00:00")
    assert emerging_tactics[0]["share"] == pytest.approx(1 / 3)

    emerging_experiments = digest["emerging_experiments"]
    assert emerging_experiments[0]["experiment_id"] == "exp-reversion"
    assert emerging_experiments[0]["first_seen"].endswith("+00:00")

    tag_entries = digest["tags"]
    assert [entry["tag"] for entry in tag_entries[:2]] == ["fast-weight", "momentum"]


def test_exploration_budget_enforces_flow_limit() -> None:
    router = PolicyRouter(exploration_max_fraction=0.25)
    router.register_tactic(
        PolicyTactic(
            tactic_id="core",
            base_weight=1.0,
            regime_bias={"bull": 1.0},
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="explore",
            base_weight=1.4,
            regime_bias={"bull": 1.0},
            exploration=True,
            tags=("exploration",),
        )
    )

    base_time = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

    for offset in range(3):
        decision = router.route(_regime(timestamp=base_time + timedelta(minutes=offset)))
        assert decision.tactic_id == "core"
        blocked = decision.exploration_metadata.get("blocked_candidates", [])
        assert blocked and blocked[0]["reason"] == "budget_exhausted"

    allowed_decision = router.route(_regime(timestamp=base_time + timedelta(minutes=3)))
    assert allowed_decision.tactic_id == "explore"
    metadata = allowed_decision.exploration_metadata
    assert metadata["selected_is_exploration"] is True
    assert metadata["budget_before"]["total_decisions"] == 3
    assert metadata["budget_after"]["exploration_decisions"] == 1

    follow_up = router.route(_regime(timestamp=base_time + timedelta(minutes=4)))
    assert follow_up.tactic_id == "core"
    follow_blocked = follow_up.exploration_metadata.get("blocked_candidates", [])
    assert follow_blocked and follow_blocked[0]["reason"] == "budget_exhausted"


def test_exploration_budget_respects_mutation_cadence() -> None:
    router = PolicyRouter(exploration_mutate_every=3)
    router.register_tactic(
        PolicyTactic(
            tactic_id="core",
            base_weight=1.0,
            regime_bias={"bull": 1.0},
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="explore",
            base_weight=1.3,
            regime_bias={"bull": 1.0},
            exploration=True,
            tags=("exploration",),
        )
    )

    base_time = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

    first_decision = router.route(_regime(timestamp=base_time))
    assert first_decision.tactic_id == "explore"
    assert first_decision.exploration_metadata["selected_is_exploration"] is True

    for offset in range(1, 4):
        decision = router.route(
            _regime(timestamp=base_time + timedelta(minutes=offset))
        )
        assert decision.tactic_id == "core"
        blocked = decision.exploration_metadata.get("blocked_candidates", [])
        assert blocked and blocked[0]["reason"] == "cadence"

    follow_up = router.route(_regime(timestamp=base_time + timedelta(minutes=4)))
    assert follow_up.tactic_id == "explore"
    follow_meta = follow_up.exploration_metadata
    assert follow_meta["selected_is_exploration"] is True
    assert follow_meta["budget_after"]["exploration_decisions"] == 2


def test_exploration_freeze_blocks_and_releases() -> None:
    router = PolicyRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="core",
            base_weight=1.0,
            regime_bias={"bull": 1.0},
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="explore",
            base_weight=1.2,
            regime_bias={"bull": 1.0},
            exploration=True,
        )
    )

    initial = router.route(_regime())
    assert initial.tactic_id == "explore"

    router.freeze_exploration(
        reason="risk_breach",
        triggered_by="risk_gateway",
        severity="critical",
        metadata={"violation": "portfolio_risk_breach"},
    )

    frozen_decision = router.route(_regime())
    assert frozen_decision.tactic_id == "core"
    metadata = frozen_decision.exploration_metadata
    assert metadata.get("selected_is_exploration") is False
    freeze_state = metadata.get("freeze_state")
    assert isinstance(freeze_state, Mapping)
    assert freeze_state.get("active") is True
    blocked = metadata.get("blocked_candidates", [])
    assert blocked and blocked[0].get("reason") == "frozen"

    router.release_exploration(reason="stability_recovered")
    recovered = router.route(_regime())
    assert recovered.tactic_id == "explore"
    post_state = recovered.exploration_metadata.get("freeze_state")
    if post_state is not None:
        assert post_state.get("active") is False


def test_tournament_selection_requires_history() -> None:
    router = PolicyRouter(tournament_size=3, tournament_min_regime_decisions=5)
    router.register_tactics(
        (
            PolicyTactic(tactic_id="alpha", base_weight=1.0, regime_bias={"bull": 1.05}),
            PolicyTactic(tactic_id="beta", base_weight=0.95, regime_bias={"bull": 1.02}),
        )
    )

    decision = router.route(_regime(regime="bull"))

    summary = decision.reflection_summary.get("tournament_selection")
    assert summary is not None
    assert summary["reason"] == "insufficient_history"
    assert summary["decisions_observed"] == 0
    assert summary["tournament_size"] == 3


def test_regime_fitness_tournament_promotes_global_performer() -> None:
    router = PolicyRouter(
        tournament_size=3,
        tournament_min_regime_decisions=2,
        tournament_weights={
            "current": 0.1,
            "regime": 0.2,
            "global": 0.55,
            "regime_coverage": 0.15,
        },
        tournament_bonus=1.0,
    )
    router.register_tactics(
        (
            PolicyTactic(
                tactic_id="alpha",
                base_weight=1.0,
                regime_bias={"bull": 1.05, "bear": 0.7},
                confidence_sensitivity=0.0,
            ),
            PolicyTactic(
                tactic_id="beta",
                base_weight=0.97,
                regime_bias={"bull": 1.04, "bear": 1.5},
                confidence_sensitivity=0.0,
            ),
        )
    )

    base_time = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

    for offset in range(3):
        router.route(
            RegimeState(
                regime="bear",
                confidence=0.82,
                features={"volume_z": 0.1, "volatility": 0.35},
                timestamp=base_time + timedelta(minutes=offset),
            )
        )

    for offset in range(2):
        router.route(
            RegimeState(
                regime="bull",
                confidence=0.78,
                features={"volume_z": 0.14, "volatility": 0.22},
                timestamp=base_time + timedelta(hours=1, minutes=offset),
            )
        )

    final_decision = router.route(
        RegimeState(
            regime="bull",
            confidence=0.8,
            features={"volume_z": 0.2, "volatility": 0.2},
            timestamp=base_time + timedelta(hours=1, minutes=10),
        )
    )

    assert final_decision.tactic_id == "beta"

    tournament = final_decision.reflection_summary.get("tournament_selection")
    assert tournament is not None
    assert tournament["reason"] == "tournament"
    assert tournament["winner_reason"] == "composite_bonus"
    assert tournament["selected_tactic"] == "beta"
    candidate_ids = {candidate["tactic_id"] for candidate in tournament["candidates"]}
    assert {"alpha", "beta"}.issubset(candidate_ids)

    snapshot = router.regime_fitness_snapshot()
    assert snapshot["regimes"]["bear"]["decisions"] == 3
    beta_regimes = snapshot["tactics"]["beta"]["regimes"]
    assert beta_regimes["bear"]["observations"] >= 3
    assert beta_regimes["bull"]["observations"] >= 3


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
