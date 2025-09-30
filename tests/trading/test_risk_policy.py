from __future__ import annotations

from decimal import Decimal
from typing import Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.risk.risk_policy import RiskPolicy


def _state(open_positions: Mapping[str, Mapping[str, float]] | None = None) -> Mapping[str, object]:
    return {
        "equity": 100_000.0,
        "open_positions": open_positions or {},
        "current_daily_drawdown": 0.02,
    }


def test_risk_policy_approves_within_limits() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1000.0,
        price=1.1,
        stop_loss_pct=0.02,
        portfolio_state=_state(),
    )

    assert decision.approved
    assert not decision.violations
    assert decision.metadata["projected_total_exposure"] > 0


def test_risk_policy_rejects_when_exposure_exceeds_limit() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.1"),
        max_leverage=Decimal("1.5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = _state(
        {
            "EURUSD": {"quantity": 5000.0, "last_price": 1.2},
            "GBPUSD": {"quantity": 4000.0, "last_price": 1.3},
        }
    )

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=6000.0,
        price=1.2,
        stop_loss_pct=0.03,
        portfolio_state=state,
    )

    assert not decision.approved
    assert "policy.max_total_exposure_pct" in decision.violations
    assert decision.metadata["projected_total_exposure"] > decision.metadata["max_total_exposure"]


def test_risk_policy_allows_closing_position() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.4"),
        max_leverage=Decimal("4.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = _state({"EURUSD": {"quantity": 5000.0, "last_price": 1.1}})

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=-5000.0,
        price=1.1,
        stop_loss_pct=0.02,
        portfolio_state=state,
    )

    assert decision.approved
    assert not decision.violations
    assert decision.metadata["exposure_increase"] == pytest.approx(0.0)


def test_risk_policy_warns_but_allows_in_research_mode() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.01"),
        max_total_exposure_pct=Decimal("0.1"),
        max_leverage=Decimal("1.2"),
        max_drawdown_pct=Decimal("0.05"),
        min_position_size=1,
        research_mode=True,
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=10_000.0,
        price=2.0,
        stop_loss_pct=0.05,
        portfolio_state=_state(),
    )

    assert decision.approved
    assert decision.violations
    assert decision.metadata["research_mode"] is True


def test_risk_policy_resolves_price_from_open_position() -> None:
    """Ensure we reuse canonical position pricing when no quote is supplied."""

    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        mandatory_stop_loss=True,
    )
    policy = RiskPolicy.from_config(config)

    state = _state({"EURUSD": {"quantity": 2500.0, "avg_price": 1.15}})

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1000.0,
        price=0.0,
        stop_loss_pct=0.02,
        portfolio_state=state,
    )

    assert decision.approved
    assert not decision.violations
    assert decision.metadata["resolved_price"] == pytest.approx(1.15)


def test_risk_policy_requires_stop_loss_when_mandated() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        mandatory_stop_loss=True,
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1000.0,
        price=1.2,
        stop_loss_pct=0.0,
        portfolio_state=_state(),
    )

    assert not decision.approved
    assert decision.reason == "policy.stop_loss"
    assert "policy.stop_loss" in decision.violations


def test_risk_policy_derives_equity_from_cash_and_positions() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = {
        "equity": 0.0,
        "cash": 25_000.0,
        "open_positions": {
            "EURUSD": {"quantity": 1_000.0, "last_price": 1.1},
        },
        "current_daily_drawdown": 0.01,
    }

    decision = policy.evaluate(
        symbol="GBPUSD",
        quantity=500.0,
        price=1.2,
        stop_loss_pct=0.02,
        portfolio_state=state,
    )

    assert decision.approved
    assert decision.metadata["equity"] == pytest.approx(26_100.0)
    assert decision.metadata["risk_budget"] == pytest.approx(522.0)
