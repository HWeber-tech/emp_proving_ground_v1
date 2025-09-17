"""Regression tests for src.risk.risk_manager_impl.RiskManagerImpl."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

from src.risk.risk_manager_impl import RiskManagerImpl


@pytest.mark.asyncio
async def test_validate_position_rejects_invalid_inputs() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)

    invalid_size = await manager.validate_position(
        {"symbol": "EURUSD", "size": 0, "entry_price": 1.1}
    )
    invalid_price = await manager.validate_position(
        {"symbol": "EURUSD", "size": 1_000, "entry_price": 0}
    )

    assert invalid_size is False
    assert invalid_price is False


@pytest.mark.asyncio
async def test_validate_position_enforces_max_risk() -> None:
    manager = RiskManagerImpl(initial_balance=5_000)

    within_limits = await manager.validate_position(
        {"symbol": "EURUSD", "size": 4_000, "entry_price": 1.1}
    )
    breaching_limits = await manager.validate_position(
        {"symbol": "EURUSD", "size": 6_000, "entry_price": 1.1}
    )

    assert within_limits is True
    assert breaching_limits is False


@pytest.mark.asyncio
async def test_calculate_position_size_applies_kelly_fraction() -> None:
    manager = RiskManagerImpl(initial_balance=20_000)
    signal = {"symbol": "EURUSD", "confidence": 0.7, "stop_loss_pct": 0.02}

    size = await manager.calculate_position_size(signal)

    # Kelly fraction resolves to 0.55 which yields 11_000 given the balance and risk caps.
    assert pytest.approx(11_000.0, rel=1e-6) == size


@pytest.mark.asyncio
async def test_calculate_position_size_throttles_during_drawdown_and_recovers() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    signal = {"symbol": "EURUSD", "confidence": 0.65, "stop_loss_pct": 0.01}

    baseline = await manager.calculate_position_size(signal)
    manager.update_account_balance(7_500)  # Trigger max configured drawdown.
    drawdown = await manager.calculate_position_size(signal)
    manager.update_account_balance(11_000)  # New peak should restore full risk budget.
    recovery = await manager.calculate_position_size(signal)

    assert baseline == pytest.approx(9_500.0)
    assert drawdown == pytest.approx(1_781.25, rel=1e-6)
    assert recovery == pytest.approx(10_450.0)


@pytest.mark.asyncio
async def test_calculate_position_size_respects_updated_risk_limits() -> None:
    manager = RiskManagerImpl(initial_balance=15_000)
    signal = {"symbol": "EURUSD", "confidence": 0.7, "stop_loss_pct": 0.02}

    baseline = await manager.calculate_position_size(signal)
    manager.update_limits({"max_position_risk": 0.01})
    constrained = await manager.calculate_position_size(signal)

    assert baseline == pytest.approx(8_250.0)
    assert constrained == pytest.approx(4_125.0)


@pytest.mark.asyncio
async def test_calculate_position_size_handles_exception() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)

    class CrashingSignal(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - delegated to trigger crash
            raise RuntimeError("boom")

    size = await manager.calculate_position_size(CrashingSignal())

    assert size == 1_000.0


def test_update_limits_accepts_decimal_inputs() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)

    manager.update_limits({"max_position_risk": Decimal("0.05"), "max_drawdown": Decimal("0.4")})

    assert manager.config.max_position_risk == pytest.approx(0.05)
    assert manager.config.max_drawdown == pytest.approx(0.4)


def test_evaluate_portfolio_risk_converts_values(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    captured: Dict[str, Dict[str, float]] = {}

    def fake_assess(positions: Dict[str, float]) -> float:
        captured["positions"] = positions
        return 0.42

    monkeypatch.setattr(manager.risk_manager, "assess_risk", fake_assess)

    risk = manager.evaluate_portfolio_risk({"EURUSD": Decimal("1.25"), "GBPUSD": Decimal("2.5")})

    assert captured["positions"] == {"EURUSD": 1.25, "GBPUSD": 2.5}
    assert risk == pytest.approx(0.42)


def test_get_risk_summary_reports_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    manager.add_position("EURUSD", 1_000, 1.1)
    manager.update_position_value("EURUSD", 1.2)

    monkeypatch.setattr(
        manager.risk_manager, "assess_risk", lambda positions: {"total": sum(positions.values())}
    )

    summary = manager.get_risk_summary()

    assert summary["account_balance"] == pytest.approx(10_000.0)
    assert summary["positions"] == pytest.approx(1.0)
    assert summary["tracked_positions"] == ["EURUSD"]
    assert summary["assessed_risk"] == {"total": 1_000.0}


def test_calculate_portfolio_risk_aggregates_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    manager.add_position("EURUSD", 1_500, 1.1)
    manager.add_position("GBPUSD", 500, 1.2)

    monkeypatch.setattr(
        manager.risk_manager, "assess_risk", lambda positions: sum(positions.values()) * 0.1
    )

    snapshot = manager.calculate_portfolio_risk()

    assert snapshot["total_size"] == pytest.approx(2_000.0)
    assert snapshot["risk_amount"] == pytest.approx(40.0)  # 2% of each position size aggregated
    assert snapshot["assessed_risk"] == pytest.approx(200.0)


def test_get_position_risk_handles_unknown_symbol() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)

    assert manager.get_position_risk("EURUSD") == {}


def test_get_position_risk_reports_tracked_position() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    manager.add_position("EURUSD", 1_000, 1.1)
    manager.update_position_value("EURUSD", 1.3)

    metrics = manager.get_position_risk("EURUSD")

    assert metrics["symbol"] == "EURUSD"
    assert metrics["size"] == pytest.approx(1_000.0)
    assert metrics["entry_price"] == pytest.approx(1.1)
    assert metrics["current_price"] == pytest.approx(1.3)
    assert metrics["risk_amount"] == pytest.approx(20.0)


def test_propose_rebalance_returns_copy() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)
    positions = {"EURUSD": 1_000.0, "GBPUSD": 500.0}

    rebalance = manager.propose_rebalance(positions)

    assert rebalance == positions
    assert rebalance is not positions


@pytest.mark.asyncio
async def test_validate_position_handles_exception() -> None:
    manager = RiskManagerImpl(initial_balance=10_000)

    class CrashingMapping(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - delegated to trigger crash
            raise RuntimeError("boom")

    result = await manager.validate_position(CrashingMapping())

    assert result is False
