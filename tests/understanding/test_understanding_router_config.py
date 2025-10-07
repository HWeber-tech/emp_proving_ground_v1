from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.understanding.router_config import UnderstandingRouterConfig


pytestmark = pytest.mark.guardrail


def test_router_config_builds_adapters_with_hebbian() -> None:
    config = UnderstandingRouterConfig.from_mapping(
        {
            "feature_flag": "fast_weights_live",
            "default_fast_weights_enabled": False,
            "adapters": [
                {
                    "adapter_id": "liquidity_rescue",
                    "tactic_id": "alpha_strike",
                    "rationale": "Lean into alpha tactic when liquidity stressed",
                    "multiplier": 1.25,
                    "feature_gates": [
                        {"feature": "liquidity_z", "maximum": -0.3},
                    ],
                    "required_flags": {"fast_weights_live": True},
                },
                {
                    "adapter_id": "momentum_boost",
                    "tactic_id": "alpha_strike",
                    "rationale": "Hebbian-style momentum boost",
                    "expires_at": datetime(2024, 2, 1, tzinfo=timezone.utc).isoformat(),
                    "hebbian": {
                        "feature": "momentum",
                        "learning_rate": 0.4,
                        "decay": 0.15,
                        "baseline": 1.0,
                        "floor": 0.2,
                        "ceiling": 2.5,
                    },
                },
            ],
        }
    )

    adapters = config.build_adapters()
    assert set(adapters) == {"liquidity_rescue", "momentum_boost"}

    static_adapter = adapters["liquidity_rescue"]
    assert static_adapter.multiplier == pytest.approx(1.25)
    assert static_adapter.feature_gates[0].maximum == pytest.approx(-0.3)

    hebbian_adapter = adapters["momentum_boost"]
    assert hebbian_adapter.hebbian is not None
    assert hebbian_adapter.hebbian.learning_rate == pytest.approx(0.4)
    assert hebbian_adapter.expires_at == datetime(2024, 2, 1, tzinfo=timezone.utc)


def test_router_config_tier_defaults_with_fallback() -> None:
    config = UnderstandingRouterConfig.from_mapping(
        {
            "default_fast_weights_enabled": False,
            "tier_defaults": {
                "bootstrap": {
                    "fast_weights_enabled": False,
                    "enabled_adapters": ["liquidity_rescue"],
                },
                "institutional": {
                    "fast_weights_enabled": True,
                    "enabled_adapters": ["liquidity_rescue", "momentum_boost"],
                },
            },
        }
    )

    bootstrap = config.for_tier("Bootstrap")
    assert bootstrap.fast_weights_enabled is False
    assert bootstrap.enabled_adapters == ("liquidity_rescue",)

    institutional = config.for_tier("institutional")
    assert institutional.fast_weights_enabled is True
    assert institutional.enabled_adapters == ("liquidity_rescue", "momentum_boost")

    fallback = config.for_tier(None)
    assert fallback.fast_weights_enabled is False
