from __future__ import annotations

import math
from decimal import Decimal

import pytest

from src.risk.real_risk_manager import RealRiskConfig, RealRiskManager


def test_real_risk_config_normalizes_negative_limits() -> None:
    config = RealRiskConfig(
        max_position_risk=-1.0,
        max_total_exposure=0.0,
        max_drawdown=0.4,
        max_leverage=-2.0,
        equity=-10.0,
    )

    assert config.max_total_exposure == pytest.approx(0.4)
    assert config.max_position_risk == pytest.approx(0.02)
    assert config.max_leverage == pytest.approx(10.0)
    assert config.equity == pytest.approx(0.0)


def test_real_risk_manager_update_equity_handles_invalid_input() -> None:
    config = RealRiskConfig(equity=5_000.0)
    manager = RealRiskManager(config)

    manager.update_equity("not-a-number")

    assert manager.equity == pytest.approx(5_000.0)


def test_real_risk_manager_assess_risk_ignores_non_finite_values() -> None:
    manager = RealRiskManager(RealRiskConfig())

    score = manager.assess_risk(
        {
            "EURUSD": float("nan"),
            "GBPUSD": float("inf"),
            "AUDUSD": 2_500.0,
        }
    )

    assert math.isfinite(score)
    assert score > 0
    snapshot = manager.last_snapshot
    assert snapshot["max_exposure"] > 0


def test_real_risk_manager_resolve_budget_fallback_paths() -> None:
    # Percent * equity <= 0 triggers fallback resolution order.
    assert RealRiskManager._resolve_budget(0.0, 10_000.0, 5_000.0) == pytest.approx(10_000.0)
    assert RealRiskManager._resolve_budget(0.0, 0.0, 5_000.0) == pytest.approx(5_000.0)
    assert RealRiskManager._resolve_budget(0.0, 0.0, 0.0) == pytest.approx(1.0)
