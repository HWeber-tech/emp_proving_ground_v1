from __future__ import annotations

import pytest

from src.governance.system_config import EmpEnvironment, EmpTier, RunMode, SystemConfig
from src.thinking.adaptation.feature_toggles import (
    AdaptationFeatureToggles,
    EXPLORATION_FLAG,
    FAST_WEIGHTS_FLAG,
    LINEAR_ATTENTION_FLAG,
)


def _config(
    *,
    environment: EmpEnvironment = EmpEnvironment.demo,
    run_mode: RunMode = RunMode.paper,
    tier: EmpTier = EmpTier.tier_1,
    extras: dict[str, str] | None = None,
) -> SystemConfig:
    return SystemConfig(
        run_mode=run_mode,
        environment=environment,
        tier=tier,
        extras=extras or {},
    )


def test_demo_environment_defaults_disable_features() -> None:
    config = _config(environment=EmpEnvironment.demo, tier=EmpTier.tier_0)
    toggles = AdaptationFeatureToggles.from_system_config(config)

    assert toggles.fast_weights is False
    assert toggles.linear_attention is False
    assert toggles.exploration is False


def test_staging_enables_adaptation_when_not_live() -> None:
    config = _config(environment=EmpEnvironment.staging, tier=EmpTier.tier_1)
    toggles = AdaptationFeatureToggles.from_system_config(config)

    assert toggles.fast_weights is True
    assert toggles.linear_attention is True
    assert toggles.exploration is False


def test_live_run_mode_forces_conservative_posture() -> None:
    config = _config(
        environment=EmpEnvironment.staging,
        run_mode=RunMode.live,
        extras={
            "FEATURE_FAST_WEIGHTS": "yes",
            "FEATURE_LINEAR_ATTENTION": "on",
            "FEATURE_EXPLORATION": "true",
        },
    )
    toggles = AdaptationFeatureToggles.from_system_config(config)

    assert toggles.fast_weights is False
    assert toggles.linear_attention is False
    assert toggles.exploration is False


@pytest.mark.parametrize(
    "key, field",
    [
        ("FEATURE_FAST_WEIGHTS", FAST_WEIGHTS_FLAG),
        ("FEATURE_LINEAR_ATTENTION", LINEAR_ATTENTION_FLAG),
        ("FEATURE_EXPLORATION", EXPLORATION_FLAG),
    ],
)
def test_extras_override_environment_defaults(key: str, field: str) -> None:
    config = _config(environment=EmpEnvironment.demo, extras={key: "yes"})
    toggles = AdaptationFeatureToggles.from_system_config(config)

    flags = toggles.as_feature_flags()
    assert flags[field] is True


def test_merge_flags_preserves_snapshot_overrides() -> None:
    toggles = AdaptationFeatureToggles(fast_weights=False, linear_attention=False)
    snapshot = {FAST_WEIGHTS_FLAG: True}

    merged = toggles.merge_flags(snapshot)

    assert merged[FAST_WEIGHTS_FLAG] is True
    assert merged[LINEAR_ATTENTION_FLAG] is False


def test_fast_weight_resolution_prefers_explicit_hint() -> None:
    toggles = AdaptationFeatureToggles(fast_weights=False)
    assert toggles.resolve_fast_weights_enabled(True) is True
    assert toggles.resolve_fast_weights_enabled(None) is False
