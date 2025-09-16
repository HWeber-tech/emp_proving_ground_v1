from __future__ import annotations

from src.governance.system_config import (
    ConnectionProtocol,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)


def test_with_updated_accepts_strings_and_copies_extras() -> None:
    base = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_0,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.fix,
        extras={"legacy": "value"},
    )

    updated = base.with_updated(
        run_mode="LIVE",
        environment="production",
        tier="tier-2",
        confirm_live="yes",
        connection_protocol="fix",
        extras={"new": "1"},
    )

    assert updated is not base
    assert updated.run_mode is RunMode.live
    assert updated.environment is EmpEnvironment.production
    assert updated.tier is EmpTier.tier_2
    assert updated.confirm_live is True
    assert updated.connection_protocol is ConnectionProtocol.fix
    assert updated.extras == {"new": "1"}
    assert base.extras == {"legacy": "value"}


def test_with_updated_invalid_values_fallback_to_existing() -> None:
    base = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_1,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.fix,
    )

    updated = base.with_updated(run_mode="??", confirm_live="maybe", connection_protocol="http")

    assert updated.run_mode is base.run_mode
    assert updated.confirm_live is base.confirm_live
    assert updated.connection_protocol is base.connection_protocol
    assert updated.environment is base.environment
    assert updated.tier is base.tier
    assert updated.extras == base.extras


def test_with_updated_accepts_enum_instances() -> None:
    base = SystemConfig()

    updated = base.with_updated(
        run_mode=RunMode.live,
        environment=EmpEnvironment.staging,
        tier=EmpTier.tier_2,
        confirm_live=True,
        connection_protocol=ConnectionProtocol.fix,
    )

    assert updated.run_mode is RunMode.live
    assert updated.environment is EmpEnvironment.staging
    assert updated.tier is EmpTier.tier_2
    assert updated.confirm_live is True
    assert updated.connection_protocol is ConnectionProtocol.fix

    # Original instance remains unchanged
    assert base.run_mode is RunMode.paper
    assert base.environment is EmpEnvironment.demo
    assert base.tier is EmpTier.tier_0
    assert base.confirm_live is False
