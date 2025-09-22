"""Unit tests for :mod:`src.risk.real_risk_manager`."""

from __future__ import annotations

import pytest

from src.risk.real_risk_manager import RealRiskConfig, RealRiskManager


def test_assess_risk_returns_zero_for_empty_positions() -> None:
    manager = RealRiskManager(RealRiskConfig())

    score = manager.assess_risk({})

    assert score == pytest.approx(0.0)
    assert manager.last_snapshot["risk_score"] == pytest.approx(0.0)
    assert manager.last_snapshot["total_exposure"] == pytest.approx(0.0)


def test_assess_risk_calculates_ratios() -> None:
    config = RealRiskConfig(max_position_risk=0.02, max_drawdown=0.25, equity=50_000)
    manager = RealRiskManager(config)

    score = manager.assess_risk({"EURUSD": 1_000, "GBPUSD": 500})
    snapshot = manager.last_snapshot

    assert score == pytest.approx(1.0)
    assert snapshot["position_ratio"] == pytest.approx(1.0)
    assert snapshot["total_ratio"] == pytest.approx(0.12)
    assert snapshot["gross_leverage"] == pytest.approx(0.03)
    assert snapshot["leverage_ratio"] == pytest.approx(0.003)


def test_update_equity_rebalances_budgets() -> None:
    manager = RealRiskManager(RealRiskConfig(equity=10_000))

    initial = manager.assess_risk({"EURUSD": 1_000})
    manager.update_equity(50_000)
    updated = manager.assess_risk({"EURUSD": 1_000})

    assert initial == pytest.approx(5.0)
    assert updated == pytest.approx(1.0)


def test_assess_risk_ignores_non_numeric_positions() -> None:
    manager = RealRiskManager(RealRiskConfig(equity=20_000))

    score = manager.assess_risk({"EURUSD": float("nan"), "GBPUSD": "oops", "USDJPY": 250})

    # Only the valid USDJPY position should contribute to the risk calculation.
    assert score == pytest.approx(250 / (0.02 * 20_000))


def test_assess_risk_flags_leverage_breaches() -> None:
    config = RealRiskConfig(
        max_position_risk=5.0,
        max_total_exposure=5.0,
        max_drawdown=5.0,
        max_leverage=2.0,
        equity=10_000,
    )
    manager = RealRiskManager(config)

    score = manager.assess_risk({"EURUSD": 30_000})
    snapshot = manager.last_snapshot

    assert score == pytest.approx(1.5)
    assert snapshot["leverage_ratio"] == pytest.approx(1.5)
