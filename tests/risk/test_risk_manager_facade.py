"""Regression coverage for the ``src.risk.manager`` facade."""

from __future__ import annotations

from decimal import Decimal

import pytest

import src.core  # noqa: F401  # Prime lazy exports to avoid circular imports.
from src.config.risk.risk_config import RiskConfig
from src.risk import RiskManager


@pytest.fixture()
def base_config() -> RiskConfig:
    """Provide a permissive risk configuration for facade tests."""

    return RiskConfig(
        min_position_size=1,
        max_position_size=100000,
    )


def test_risk_manager_requires_config() -> None:
    """Constructing the facade without a configuration should fail fast."""

    with pytest.raises(ValueError):
        RiskManager()


def test_risk_manager_rejects_non_mapping_config_payload() -> None:
    """Arbitrary payloads are rejected with a clear ``TypeError``."""

    with pytest.raises(TypeError):
        RiskManager(config=object())


def test_risk_manager_validates_mapping_payload() -> None:
    """Invalid mapping payloads surface as ``ValueError`` from ``RiskConfig`` parsing."""

    with pytest.raises(ValueError):
        RiskManager(config={"max_risk_per_trade_pct": Decimal("2.0")})


def test_risk_manager_accepts_riskconfig_instance(base_config: RiskConfig) -> None:
    """Happy-path validation permits trades that honour all configured limits."""

    manager = RiskManager(config=base_config, initial_balance=Decimal("10000"))

    assert manager.validate_trade(
        Decimal("10"),
        Decimal("100"),
        symbol="EURUSD",
        stop_loss_pct=Decimal("0.02"),
    )


def test_risk_manager_blocks_missing_stop_loss(base_config: RiskConfig) -> None:
    """Mandatory stop-loss enforcement remains in place on the facade."""

    manager = RiskManager(config=base_config, initial_balance=Decimal("10000"))

    assert not manager.validate_trade(
        Decimal("10"),
        Decimal("100"),
        symbol="EURUSD",
    )


def test_risk_manager_enforces_sector_budget() -> None:
    """Sector allocations honour the configured exposure budget from ``RiskConfig``."""

    config = RiskConfig(
        min_position_size=1,
        max_position_size=100000,
        max_risk_per_trade_pct=Decimal("0.05"),
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.01")},
    )
    manager = RiskManager(config=config, initial_balance=Decimal("10000"))

    assert not manager.validate_trade(
        Decimal("1000"),
        Decimal("1"),
        symbol="EURUSD",
        stop_loss_pct=Decimal("0.2"),
    )
