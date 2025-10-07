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
    UnderstandingRouter,
)


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
        features=regime_state.features,
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
    assert "expires_at" not in summary


def test_understanding_router_respects_fast_weight_disable_flag() -> None:
    router = _build_router()
    snapshot = _build_snapshot(fast_weights_enabled=False)

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.applied_adapters == ()
    assert decision_bundle.fast_weight_summary == {}


def test_understanding_router_feature_gate_blocks_without_threshold() -> None:
    router = _build_router()
    snapshot = _build_snapshot(liquidity_z=0.1)

    decision_bundle = router.route(snapshot)

    assert decision_bundle.decision.tactic_id == "beta_hold"
    assert decision_bundle.applied_adapters == ()


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
