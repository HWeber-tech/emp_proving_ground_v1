from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)


def test_system_config_env_roundtrip_valid() -> None:
    env = {
        "RUN_MODE": "live",
        "EMP_ENVIRONMENT": "staging",
        "EMP_TIER": "tier-1",  # alias accepted
        "CONFIRM_LIVE": "yes",
        "CONNECTION_PROTOCOL": "fix",
        "DATA_BACKBONE_MODE": "institutional",
        "FOO": "bar",  # unknown should land in extras
    }
    cfg = SystemConfig.from_env(env)
    assert cfg.run_mode == RunMode.live
    assert cfg.environment == EmpEnvironment.staging
    assert cfg.tier == EmpTier.tier_1
    assert cfg.confirm_live is True
    assert cfg.connection_protocol == ConnectionProtocol.fix
    assert cfg.data_backbone_mode is DataBackboneMode.institutional

    # Backward-compatible string views
    assert cfg.run_mode_str == "live"
    assert cfg.environment_str == "staging"
    assert cfg.tier_str == "tier_1"
    assert cfg.connection_protocol_str == "fix"
    assert cfg.data_backbone_mode_str == "institutional"

    # Extras should retain unknown keys
    assert "FOO" in cfg.extras and cfg.extras["FOO"] == "bar"

    # to_env / from_env preserves primary fields
    cfg2 = SystemConfig.from_env(cfg.to_env())
    assert cfg2.run_mode == cfg.run_mode
    assert cfg2.environment == cfg.environment
    assert cfg2.tier == cfg.tier
    assert cfg2.confirm_live == cfg.confirm_live
    assert cfg2.connection_protocol == cfg.connection_protocol
    assert cfg2.data_backbone_mode == cfg.data_backbone_mode


def test_system_config_invalid_tracking() -> None:
    env = {
        "RUN_MODE": "super",  # invalid
        "CONFIRM_LIVE": "maybe",  # invalid
        "DATA_BACKBONE_MODE": "other",  # invalid
    }
    cfg = SystemConfig.from_env(env)

    # Should fall back to defaults
    assert cfg.run_mode == RunMode.paper
    assert cfg.confirm_live is False
    assert cfg.data_backbone_mode is DataBackboneMode.bootstrap

    # Invalid markers captured in extras
    assert "RUN_MODE_invalid" in cfg.extras and cfg.extras["RUN_MODE_invalid"] == "super"
    assert "CONFIRM_LIVE_invalid" in cfg.extras and cfg.extras["CONFIRM_LIVE_invalid"] == "maybe"
    assert (
        "DATA_BACKBONE_MODE_invalid" in cfg.extras
        and cfg.extras["DATA_BACKBONE_MODE_invalid"] == "other"
    )

    # Environment not provided - default remains
    assert cfg.environment == EmpEnvironment.demo
