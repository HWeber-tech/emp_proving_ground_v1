"""Tests for typed, Enum-based SystemConfig with safe env coercion."""
from __future__ import annotations

from src.governance.system_config import (
    ConnectionProtocol,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)


def test_defaults_match_ci():
    cfg = SystemConfig.from_env({})
    assert cfg.run_mode is RunMode.paper
    assert cfg.environment is EmpEnvironment.demo
    assert cfg.tier is EmpTier.tier_0
    assert cfg.connection_protocol is ConnectionProtocol.fix
    assert cfg.confirm_live is False


def test_case_insensitive_and_aliases():
    env = {
        "RUN_MODE": "PAPER",
        "EMP_ENVIRONMENT": "DEMO",
        "EMP_TIER": "Tier-1",
        "CONFIRM_LIVE": "True",
        "CONNECTION_PROTOCOL": "FIX",
    }
    cfg = SystemConfig.from_env(env)
    assert cfg.run_mode is RunMode.paper
    assert cfg.environment is EmpEnvironment.demo
    assert cfg.tier is EmpTier.tier_1
    assert cfg.confirm_live is True
    assert cfg.connection_protocol is ConnectionProtocol.fix


def test_bad_values_fallback():
    env = {
        "RUN_MODE": "blargh",
        "EMP_ENVIRONMENT": "unknown",
        "EMP_TIER": "TIER_9",
        "CONFIRM_LIVE": "notabool",
        "CONNECTION_PROTOCOL": "http",
        "OTHER_KEY": "preserve",
    }
    defaults = SystemConfig()
    cfg = SystemConfig.from_env(env, defaults=defaults)
    # Fallbacks to defaults
    assert cfg.run_mode is defaults.run_mode
    assert cfg.environment is defaults.environment
    assert cfg.tier is defaults.tier
    assert cfg.confirm_live is defaults.confirm_live
    assert cfg.connection_protocol is defaults.connection_protocol
    # Extras contain invalid markers with raw values
    assert cfg.extras["RUN_MODE_invalid"] == "blargh"
    assert cfg.extras["EMP_ENVIRONMENT_invalid"] == "unknown"
    assert cfg.extras["EMP_TIER_invalid"] == "TIER_9"
    assert cfg.extras["CONFIRM_LIVE_invalid"] == "notabool"
    assert cfg.extras["CONNECTION_PROTOCOL_invalid"] == "http"
    # And preserve non-recognized keys
    assert cfg.extras["OTHER_KEY"] == "preserve"


def test_to_env_roundtrip():
    cfg = SystemConfig(
        run_mode=RunMode.live,
        environment=EmpEnvironment.staging,
        tier=EmpTier.tier_2,
        confirm_live=True,
        connection_protocol=ConnectionProtocol.fix,
    )
    env = cfg.to_env()
    cfg2 = SystemConfig.from_env(env)
    assert cfg2.run_mode is RunMode.live
    assert cfg2.environment is EmpEnvironment.staging
    assert cfg2.tier is EmpTier.tier_2
    assert cfg2.confirm_live is True
    assert cfg2.connection_protocol is ConnectionProtocol.fix


def test_string_views():
    cfg = SystemConfig(
        run_mode=RunMode.mock,
        environment=EmpEnvironment.production,
        tier=EmpTier.tier_1,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.fix,
    )
    # String views should be lowercase normalized
    assert cfg.run_mode_str == "mock"
    assert cfg.environment_str == "production"
    assert cfg.tier_str == "tier_1"
    assert cfg.connection_protocol_str == "fix"