"""Tests for typed, Enum-based SystemConfig with safe env coercion."""

from __future__ import annotations

from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
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
    assert cfg.connection_protocol is ConnectionProtocol.bootstrap
    assert cfg.confirm_live is False
    assert cfg.data_backbone_mode is DataBackboneMode.bootstrap


def test_case_insensitive_and_aliases():
    env = {
        "RUN_MODE": "PAPER",
        "EMP_ENVIRONMENT": "DEMO",
        "EMP_TIER": "Tier-1",
        "CONFIRM_LIVE": "True",
        "CONNECTION_PROTOCOL": "mock",
        "DATA_BACKBONE_MODE": "INSTITUTIONAL",
    }
    cfg = SystemConfig.from_env(env)
    assert cfg.run_mode is RunMode.paper
    assert cfg.environment is EmpEnvironment.demo
    assert cfg.tier is EmpTier.tier_1
    assert cfg.confirm_live is True
    assert cfg.connection_protocol is ConnectionProtocol.bootstrap
    assert cfg.data_backbone_mode is DataBackboneMode.institutional


def test_bad_values_fallback():
    env = {
        "RUN_MODE": "blargh",
        "EMP_ENVIRONMENT": "unknown",
        "EMP_TIER": "TIER_9",
        "CONFIRM_LIVE": "notabool",
        "CONNECTION_PROTOCOL": "http",
        "DATA_BACKBONE_MODE": "invalid",
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
    assert cfg.data_backbone_mode is defaults.data_backbone_mode
    # Extras contain invalid markers with raw values
    assert cfg.extras["RUN_MODE_invalid"] == "blargh"
    assert cfg.extras["EMP_ENVIRONMENT_invalid"] == "unknown"
    assert cfg.extras["EMP_TIER_invalid"] == "TIER_9"
    assert cfg.extras["CONFIRM_LIVE_invalid"] == "notabool"
    assert cfg.extras["CONNECTION_PROTOCOL_invalid"] == "http"
    assert cfg.extras["DATA_BACKBONE_MODE_invalid"] == "invalid"
    # And preserve non-recognized keys
    assert cfg.extras["OTHER_KEY"] == "preserve"


def test_to_env_roundtrip():
    cfg = SystemConfig(
        run_mode=RunMode.live,
        environment=EmpEnvironment.staging,
        tier=EmpTier.tier_2,
        confirm_live=True,
        connection_protocol=ConnectionProtocol.paper,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    env = cfg.to_env()
    cfg2 = SystemConfig.from_env(env)
    assert cfg2.run_mode is RunMode.live
    assert cfg2.environment is EmpEnvironment.staging
    assert cfg2.tier is EmpTier.tier_2
    assert cfg2.confirm_live is True
    assert cfg2.connection_protocol is ConnectionProtocol.paper
    assert cfg2.data_backbone_mode is DataBackboneMode.institutional


def test_string_views():
    cfg = SystemConfig(
        run_mode=RunMode.mock,
        environment=EmpEnvironment.production,
        tier=EmpTier.tier_1,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.fix,
        data_backbone_mode=DataBackboneMode.institutional,
    )
    # String views should be lowercase normalized
    assert cfg.run_mode_str == "mock"
    assert cfg.environment_str == "production"
    assert cfg.tier_str == "tier_1"
    assert cfg.connection_protocol_str == "fix"
    assert cfg.data_backbone_mode_str == "institutional"


def test_data_backbone_mode_infers_from_env_signals():
    env = {
        "REDIS_URL": "redis://localhost:6379/0",
        "KAFKA_BOOTSTRAP_SERVERS": "broker:9092",
    }
    cfg = SystemConfig.from_env(env)
    assert cfg.data_backbone_mode is DataBackboneMode.institutional
    # extras should still keep raw credentials for downstream use
    assert cfg.extras["REDIS_URL"] == "redis://localhost:6379/0"
    assert cfg.extras["KAFKA_BOOTSTRAP_SERVERS"] == "broker:9092"
