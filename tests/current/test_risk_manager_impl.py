"""Regression tests for src.risk.risk_manager_impl.RiskManagerImpl."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

import pytest

from src.config.risk.risk_config import RiskConfig
from src.risk import RiskManager
from src.risk.risk_manager_impl import RiskManagerImpl


def test_risk_manager_impl_requires_risk_config() -> None:
    with pytest.raises(ValueError):
        RiskManagerImpl(initial_balance=10_000, risk_config=None)


def test_risk_manager_facade_requires_risk_config() -> None:
    with pytest.raises(ValueError):
        RiskManager()


@pytest.mark.asyncio
async def test_validate_position_rejects_invalid_inputs() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

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
    manager = RiskManagerImpl(initial_balance=5_000, risk_config=RiskConfig())

    within_limits = await manager.validate_position(
        {"symbol": "EURUSD", "size": 4_000, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )
    breaching_limits = await manager.validate_position(
        {"symbol": "EURUSD", "size": 6_000, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )

    assert within_limits is True
    assert breaching_limits is False


@pytest.mark.asyncio
async def test_calculate_position_size_applies_kelly_fraction() -> None:
    manager = RiskManagerImpl(initial_balance=20_000, risk_config=RiskConfig())
    signal = {"symbol": "EURUSD", "confidence": 0.7, "stop_loss_pct": 0.02}

    size = await manager.calculate_position_size(signal)

    # Kelly fraction resolves to 0.55 which yields 11_000 given the balance and risk caps.
    assert pytest.approx(11_000.0, rel=1e-6) == size


@pytest.mark.asyncio
async def test_calculate_position_size_throttles_during_drawdown_and_recovers() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
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
    manager = RiskManagerImpl(initial_balance=15_000, risk_config=RiskConfig())
    signal = {"symbol": "EURUSD", "confidence": 0.7, "stop_loss_pct": 0.02}

    baseline = await manager.calculate_position_size(signal)
    manager.update_limits({"max_position_risk": 0.01})
    constrained = await manager.calculate_position_size(signal)

    assert baseline == pytest.approx(8_250.0)
    assert constrained == pytest.approx(4_125.0)


@pytest.mark.asyncio
async def test_calculate_position_size_handles_exception() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    class CrashingSignal(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - delegated to trigger crash
            raise RuntimeError("boom")

    size = await manager.calculate_position_size(CrashingSignal())

    assert size == pytest.approx(0.0)


def test_update_limits_accepts_decimal_inputs() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    manager.update_limits({"max_position_risk": Decimal("0.05"), "max_drawdown": Decimal("0.4")})

    assert manager.config.max_position_risk == pytest.approx(0.05)
    assert manager.config.max_drawdown == pytest.approx(0.4)
    assert manager.config.max_total_exposure == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_validate_position_requires_stop_loss() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000, "entry_price": 1.1}
    )

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_allows_research_mode_without_stop_loss() -> None:
    config = RiskConfig(research_mode=True)
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=config)

    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000, "entry_price": 1.1}
    )

    assert result is True


@pytest.mark.asyncio
async def test_update_limits_coerces_boolean_strings() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    baseline = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000, "entry_price": 1.1}
    )
    manager.update_limits({"mandatory_stop_loss": "false"})
    relaxed = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000, "entry_price": 1.1}
    )

    assert baseline is False
    assert relaxed is True


@pytest.mark.asyncio
async def test_update_limits_enables_research_mode_via_string() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    manager.update_limits({"research_mode": "true"})
    result = await manager.validate_position(
        {"symbol": "EURUSD", "size": 2_000, "entry_price": 1.1}
    )

    assert result is True


@pytest.mark.asyncio
async def test_validate_position_rejects_total_exposure_breach() -> None:
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.02"),
        max_risk_per_trade_pct=Decimal("0.015"),
        min_position_size=500,
        max_position_size=200_000,
    )
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    manager.add_position("EURUSD", 60_000, 1.1, stop_loss_pct=0.02)

    allowed = await manager.validate_position(
        {"symbol": "EURUSD", "size": 500, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )
    rejected = await manager.validate_position(
        {"symbol": "EURUSD", "size": 40_000, "entry_price": 1.1, "stop_loss_pct": 0.02}
    )

    assert allowed is True
    assert rejected is False


@pytest.mark.asyncio
async def test_calculate_position_size_respects_maximums() -> None:
    config = RiskConfig(max_position_size=5_000, min_position_size=1_000)
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    signal = {"symbol": "EURUSD", "confidence": 0.9, "stop_loss_pct": 0.01}

    size = await manager.calculate_position_size(signal)

    assert size == pytest.approx(5_000.0)


@pytest.mark.asyncio
async def test_calculate_position_size_scales_with_quantile_edge() -> None:
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=RiskConfig())
    signal = {
        "symbol": "EURUSD",
        "confidence": 0.8,
        "stop_loss_pct": 0.02,
        "quantiles": {"q25": -0.005, "q50": 0.01, "q75": 0.02},
    }

    size = await manager.calculate_position_size(signal)

    # Base size is 100k with default risk budget. Quantile ratio is 0.4 and
    # confidence scales it to 0.32 of the base size for 32k exposure.
    assert size == pytest.approx(32_000.0)


def test_evaluate_portfolio_risk_converts_values(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    captured: Dict[str, Dict[str, float]] = {}

    def fake_assess(positions: Dict[str, float]) -> float:
        captured["positions"] = positions
        return 0.42

    monkeypatch.setattr(manager.risk_manager, "assess_risk", fake_assess)

    risk = manager.evaluate_portfolio_risk({"EURUSD": Decimal("1.25"), "GBPUSD": Decimal("2.5")})

    assert captured["positions"] == {"EURUSD": 1.25, "GBPUSD": 2.5}
    assert risk == pytest.approx(0.42)


def test_get_risk_summary_reports_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 1_000, 1.1)
    manager.update_position_value("EURUSD", 1.2)

    monkeypatch.setattr(
        manager.risk_manager, "assess_risk", lambda positions: {"total": sum(positions.values())}
    )

    summary = manager.get_risk_summary()

    assert summary["account_balance"] == pytest.approx(10_000.0)
    assert summary["positions"] == pytest.approx(1.0)
    assert summary["tracked_positions"] == ["EURUSD"]
    assert summary["assessed_risk"] == {"total": 24.0}


def test_calculate_portfolio_risk_aggregates_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 1_500, 1.1)
    manager.add_position("GBPUSD", 500, 1.2)

    monkeypatch.setattr(
        manager.risk_manager, "assess_risk", lambda positions: sum(positions.values()) * 0.1
    )

    snapshot = manager.calculate_portfolio_risk()

    assert snapshot["total_size"] == pytest.approx(2_000.0)
    assert snapshot["risk_amount"] == pytest.approx(45.0)
    assert snapshot["assessed_risk"] == pytest.approx(4.5)


def test_calculate_portfolio_risk_uses_real_risk_manager_defaults() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 1_000, 1.1)

    snapshot = manager.calculate_portfolio_risk()

    # RealRiskManager should surface the per-position utilisation of the configured budgets.
    assert snapshot["risk_amount"] == pytest.approx(22.0)


def test_calculate_portfolio_risk_counts_short_exposure() -> None:
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", -10_000, 1.1, stop_loss_pct=0.02)

    snapshot = manager.calculate_portfolio_risk()

    assert snapshot["risk_amount"] == pytest.approx(220.0)
    assert snapshot["assessed_risk"] > 0.0


def test_assess_market_risk_exposes_var_and_es_metrics() -> None:
    manager = RiskManagerImpl(initial_balance=50_000, risk_config=RiskConfig())
    returns = [
        0.012,
        -0.018,
        0.007,
        -0.025,
        0.004,
        -0.011,
        0.009,
        -0.022,
    ]

    metrics = manager.assess_market_risk(returns, confidence=0.95, simulations=500)

    assert metrics["confidence"] == pytest.approx(0.95)
    assert metrics["historical_var"]["sample_size"] == pytest.approx(len(returns))
    assert metrics["monte_carlo_var"]["simulations"] == pytest.approx(500.0)
    # Expected shortfall should be at least as large as the VaR under loss framing.
    assert (
        metrics["expected_shortfall"]["historical"]["value"]
        >= metrics["historical_var"]["value"]
    )


def test_get_risk_summary_includes_market_risk_block() -> None:
    manager = RiskManagerImpl(initial_balance=25_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 2_000, 1.1)
    returns = [0.01, -0.015, 0.002, -0.02, 0.005]

    summary = manager.get_risk_summary(returns, confidence=0.9)

    assert "market_risk" in summary
    assert summary["market_risk"]["confidence"] == pytest.approx(0.9)
    assert summary["market_risk"]["historical_var"]["value"] >= 0.0


def test_calculate_portfolio_risk_embeds_market_risk_snapshot() -> None:
    manager = RiskManagerImpl(initial_balance=30_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 1_000, 1.15)
    manager.add_position("GBPUSD", 750, 1.24)
    returns = [0.006, -0.01, 0.004, -0.012, 0.003, -0.008]

    snapshot = manager.calculate_portfolio_risk(returns, confidence=0.95)

    assert "market_risk" in snapshot
    assert snapshot["market_risk"]["confidence"] == pytest.approx(0.95)
    assert snapshot["assessed_risk"] >= 0.0


def test_get_position_risk_handles_unknown_symbol() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    assert manager.get_position_risk("EURUSD") == {}


def test_get_position_risk_reports_tracked_position() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    manager.add_position("EURUSD", 1_000, 1.1)
    manager.update_position_value("EURUSD", 1.3)

    metrics = manager.get_position_risk("EURUSD")

    assert metrics["symbol"] == "EURUSD"
    assert metrics["size"] == pytest.approx(1_000.0)
    assert metrics["entry_price"] == pytest.approx(1.1)
    assert metrics["current_price"] == pytest.approx(1.3)
    assert metrics["risk_amount"] == pytest.approx(26.0)


def test_propose_rebalance_returns_copy() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())
    positions = {"EURUSD": 1_000.0, "GBPUSD": 500.0}

    rebalance = manager.propose_rebalance(positions)

    assert rebalance == positions
    assert rebalance is not positions


@pytest.mark.asyncio
async def test_validate_position_handles_exception() -> None:
    manager = RiskManagerImpl(initial_balance=10_000, risk_config=RiskConfig())

    class CrashingMapping(dict):
        def get(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - delegated to trigger crash
            raise RuntimeError("boom")

    result = await manager.validate_position(CrashingMapping())

    assert result is False


@pytest.mark.asyncio
async def test_validate_position_respects_sector_limits() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.05"),
        sector_exposure_limits={"FX": Decimal("0.01")},
        instrument_sector_map={"EURUSD": "FX", "GBPUSD": "FX"},
    )
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    manager.add_position("EURUSD", 40_000, 1.1, stop_loss_pct=0.02)

    within_limit = await manager.validate_position(
        {"symbol": "GBPUSD", "size": 5_000, "entry_price": 1.2, "stop_loss_pct": 0.02}
    )
    breaching = await manager.validate_position(
        {"symbol": "GBPUSD", "size": 10_000, "entry_price": 1.2, "stop_loss_pct": 0.02}
    )

    assert within_limit is True
    assert breaching is False


def test_check_risk_thresholds_honours_sector_budgets() -> None:
    config = RiskConfig(
        sector_exposure_limits={"FX": Decimal("0.01")},
        instrument_sector_map={"EURUSD": "FX"},
    )
    manager = RiskManagerImpl(initial_balance=100_000, risk_config=config)
    manager.add_position("EURUSD", 50_000, 1.1, stop_loss_pct=0.02)

    assert manager.check_risk_thresholds() is False

    manager.update_limits({"sector_exposure_limits": {"FX": 0.03}})
    assert manager.check_risk_thresholds() is True
