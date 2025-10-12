from __future__ import annotations

from datetime import datetime
from typing import Mapping

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback alias
    from datetime import timezone

    UTC = timezone.utc

import pytest

from src.thinking.adaptation.policy_router import PolicyTactic, RegimeState
from src.understanding.router import (
    BeliefSnapshot,
    FastWeightAdapter,
    FeatureGate,
    HebbianConfig,
    UnderstandingRouter,
)
from src.governance.system_config import SystemConfig


pytestmark = pytest.mark.guardrail


def _build_router() -> UnderstandingRouter:
    router = UnderstandingRouter()
    router.register_tactic(
        PolicyTactic(
            tactic_id="alpha_strike",
            base_weight=0.82,
            parameters={"mode": "alpha"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    router.register_tactic(
        PolicyTactic(
            tactic_id="beta_hold",
            base_weight=1.0,
            parameters={"mode": "beta"},
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    router.register_adapter(
        FastWeightAdapter(
            adapter_id="liquidity_rescue",
            tactic_id="alpha_strike",
            multiplier=1.35,
            rationale="Lean into alpha tactic when liquidity is stressed",
            feature_gates=(FeatureGate(feature="liquidity_z", maximum=-0.2),),
            required_flags={"fast_weights_live": True},
        )
    )
    return router


def _build_snapshot(
    *,
    fast_weights_enabled: bool = True,
    liquidity_z: float = -0.4,
    feature_flags: Mapping[str, bool] | None = None,
) -> BeliefSnapshot:
    regime_state = RegimeState(
        regime="balanced",
        confidence=0.5,
        features={"liquidity_z": liquidity_z},
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return BeliefSnapshot(
        belief_id="belief-balanced",
        regime_state=regime_state,
        features={"liquidity_z": liquidity_z, "momentum": 0.25},
        metadata={"window": "15m"},
        fast_weights_enabled=fast_weights_enabled,
        feature_flags=feature_flags or {"fast_weights_live": True},
    )


def test_understanding_router_applies_feature_gated_adapters() -> None:
    router = _build_router()
    snapshot = _build_snapshot()

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "alpha_strike"
    assert decision_bundle.applied_adapters == ("liquidity_rescue",)
    summary = decision_bundle.fast_weight_summary["liquidity_rescue"]
    assert summary["multiplier"] == pytest.approx(1.35)
    assert summary["feature_gates"] == [{"feature": "liquidity_z", "maximum": -0.2}]
    assert summary["required_flags"] == {"fast_weights_live": True}
    assert summary["current_multiplier"] == pytest.approx(1.35)
    assert "expires_at" not in summary
    metrics = decision_bundle.fast_weight_metrics
    assert metrics["active"] == 1
    assert metrics["total"] == 2
    assert metrics["active_percentage"] == pytest.approx(50.0)


def test_understanding_router_respects_fast_weight_disable_flag() -> None:
    router = _build_router()
    snapshot = _build_snapshot(fast_weights_enabled=False)

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.applied_adapters == ()
    assert decision_bundle.fast_weight_summary == {}
    metrics = decision_bundle.fast_weight_metrics
    assert metrics["active"] == 0
    assert metrics["total"] == 2


def test_understanding_router_ignores_external_fast_weights_when_disabled() -> None:
    router = _build_router()
    snapshot = _build_snapshot(fast_weights_enabled=False)

    decision_bundle = router.route(snapshot, fast_weights={"alpha_strike": 3.2})

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.fast_weight_summary == {}
    assert decision_bundle.fast_weight_metrics["active_percentage"] == pytest.approx(0.0)
    breakdown = decision_bundle.decision.weight_breakdown
    assert breakdown["fast_weight_multiplier"] == pytest.approx(1.0)
    assert breakdown["fast_weight_active_percentage"] == pytest.approx(0.0)


def test_understanding_router_feature_gate_blocks_without_threshold() -> None:
    router = _build_router()
    snapshot = _build_snapshot(liquidity_z=0.1)

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.applied_adapters == ()
    metrics = decision_bundle.fast_weight_metrics
    assert metrics["active"] == 0


def test_understanding_router_requires_feature_flag_enablement() -> None:
    router = _build_router()
    snapshot = _build_snapshot(feature_flags={"fast_weights_live": False})

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.applied_adapters == ()
    assert decision_bundle.fast_weight_summary == {}


def test_understanding_router_respects_adapter_expiry() -> None:
    router = _build_router()
    expiring_adapter = FastWeightAdapter(
        adapter_id="expiry_gate",
        tactic_id="alpha_strike",
        multiplier=1.2,
        rationale="Temporary boost during market opening",
        expires_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    router.register_adapter(expiring_adapter)
    snapshot = _build_snapshot()

    active_bundle = router.route(snapshot)
    assert active_bundle.applied_adapters == ("liquidity_rescue", "expiry_gate")
    assert "expiry_gate" in active_bundle.fast_weight_summary

    expired_bundle = router.route(snapshot, as_of=datetime(2024, 1, 3, tzinfo=UTC))
    assert expired_bundle.applied_adapters == ("liquidity_rescue",)
    assert "expiry_gate" not in expired_bundle.fast_weight_summary


def test_understanding_router_updates_hebbian_multiplier() -> None:
    router = _build_router()
    hebbian_adapter = FastWeightAdapter(
        adapter_id="momentum_boost",
        tactic_id="alpha_strike",
        rationale="Boost alpha when momentum is positive",
        hebbian=HebbianConfig(
            feature="momentum",
            learning_rate=0.5,
            decay=0.2,
            baseline=1.0,
            floor=0.5,
            ceiling=2.0,
        ),
    )
    router.register_adapter(hebbian_adapter)

    snapshot = _build_snapshot()

    first = router.route(snapshot)
    second = router.route(snapshot)

    first_summary = first.fast_weight_summary["momentum_boost"]
    assert first_summary["current_multiplier"] == pytest.approx(0.925)

    summary = second.fast_weight_summary["momentum_boost"]
    assert summary["previous_multiplier"] == pytest.approx(first_summary["current_multiplier"])
    assert summary["current_multiplier"] == pytest.approx(0.865)
    assert summary["current_multiplier"] <= 2.0
    assert summary["hebbian"]["feature"] == "momentum"
    assert pytest.approx(summary["feature_value"]) == 0.25

    # Disable fast weights and ensure Hebbian state does not apply.
    snapshot_disabled = _build_snapshot(fast_weights_enabled=False)
    disabled_bundle = router.route(snapshot_disabled)
    assert "momentum_boost" not in disabled_bundle.fast_weight_summary


def test_understanding_router_from_system_config_honours_fast_weight_constraints() -> None:
    config = SystemConfig(
        extras={
            "FAST_WEIGHT_MAX_ACTIVE_FRACTION": "0.2",
            "FAST_WEIGHT_ACTIVATION_THRESHOLD": "1.1",
            "FAST_WEIGHT_EXCITATORY_ONLY": "true",
        }
    )
    router = UnderstandingRouter.from_system_config(config)

    tactic_ids = ("t1", "t2", "t3", "t4", "t5")
    for identifier in tactic_ids:
        router.register_tactic(
            PolicyTactic(
                tactic_id=identifier,
                base_weight=1.0,
                parameters={"slot": identifier},
            )
        )

    snapshot = _build_snapshot()
    decision_bundle = router.route(
        snapshot,
        fast_weights={
            "t1": 1.5,
            "t2": 1.4,
            "t3": 1.2,
            "t4": 1.1,
            "t5": 0.6,
        },
    )

    assert decision_bundle.decision.tactic_id == "t1"
    metrics = decision_bundle.fast_weight_metrics
    assert metrics["total"] == 5
    assert metrics["active"] == 1
    assert metrics["active_percentage"] == pytest.approx(20.0)
    assert metrics["active_ids"] == ("t1",)
    assert metrics["min_multiplier"] == pytest.approx(1.0)
    assert metrics["max_multiplier"] == pytest.approx(1.5)
