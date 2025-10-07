"""Unit tests for the canonical portfolio monitoring configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.governance.system_config import SystemConfig
from src.trading.portfolio.config import (
    PortfolioMonitorConfig,
    resolve_portfolio_monitor_config,
)


def test_resolve_portfolio_monitor_config_defaults() -> None:
    config = resolve_portfolio_monitor_config(SystemConfig())
    assert config == PortfolioMonitorConfig()


def test_resolve_portfolio_monitor_config_with_extras() -> None:
    system_config = SystemConfig(
        extras={
            "portfolio.database_path": "/tmp/custom.db",
            "portfolio.initial_balance": "25000",
            "portfolio.save_snapshots": "false",
            "portfolio.max_total_exposure": "0.65",
        }
    )

    config = resolve_portfolio_monitor_config(system_config)

    assert config.database_path == Path("/tmp/custom.db")
    assert config.initial_balance == pytest.approx(25_000.0)
    assert config.save_snapshots is False
    assert config.max_total_exposure == pytest.approx(0.65)


def test_resolve_portfolio_monitor_config_invalid_values_fall_back() -> None:
    defaults = PortfolioMonitorConfig()
    system_config = SystemConfig(
        extras={
            "portfolio.max_positions": "0",
            "portfolio.max_position_size_pct": "2.5",
            "portfolio.max_total_exposure": "-1",
        }
    )

    config = resolve_portfolio_monitor_config(system_config)

    assert config.max_positions == defaults.max_positions
    assert config.max_position_size_pct == defaults.max_position_size_pct
    assert config.max_total_exposure == defaults.max_total_exposure


def test_resolve_portfolio_monitor_config_overrides_take_precedence() -> None:
    system_config = SystemConfig(
        extras={
            "portfolio.max_positions": "12",
        }
    )

    config = resolve_portfolio_monitor_config(
        system_config,
        overrides={"max_positions": 20, "portfolio.initial_balance": 15_000},
    )

    assert config.max_positions == 20
    assert config.initial_balance == pytest.approx(15_000.0)
