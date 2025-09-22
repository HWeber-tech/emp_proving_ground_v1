"""Integration smoke test for SystemConfig.from_env with explicit env mapping."""

from __future__ import annotations

from src.governance.system_config import SystemConfig


def test_integration_smoke_explicit_env():
    env = {
        "RUN_MODE": "paper",
        "EMP_ENVIRONMENT": "demo",
        "EMP_TIER": "tier_0",
        "CONFIRM_LIVE": "false",
        "CONNECTION_PROTOCOL": "fix",
    }
    cfg = SystemConfig.from_env(env)

    # Enums should match exactly the strings given (StrEnum)
    assert str(cfg.run_mode) == "paper"
    assert str(cfg.environment) == "demo"
    assert str(cfg.tier) == "tier_0"
    assert str(cfg.connection_protocol) == "fix"
    assert cfg.confirm_live is False
    assert str(cfg.data_backbone_mode) == "bootstrap"

    # String view properties for backward compatibility
    assert cfg.run_mode_str == "paper"
    assert cfg.environment_str == "demo"
    assert cfg.tier_str == "tier_0"
    assert cfg.connection_protocol_str == "fix"
    assert cfg.data_backbone_mode_str == "bootstrap"
