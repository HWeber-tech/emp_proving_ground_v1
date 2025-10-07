from __future__ import annotations

from decimal import Decimal
from typing import Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.risk.risk_policy import RiskPolicy


pytestmark = pytest.mark.guardrail


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


def test_risk_policy_warns_when_near_exposure_and_leverage_limits() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("0.5"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        max_position_size=50_000,
        mandatory_stop_loss=False,
    )
    policy = RiskPolicy.from_config(config)

    state = _state(
        {
            "EURUSD": {"quantity": 20_000.0, "last_price": 1.5},
        }
    )

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=10_000.0,
        price=1.5,
        stop_loss_pct=0.01,
        portfolio_state=state,
    )

    assert decision.approved is True
    assert decision.violations == tuple()

    exposure_check = next(
        check
        for check in decision.checks
        if check["name"] == "policy.max_total_exposure_pct"
    )
    assert exposure_check["status"] == "warn"
    assert exposure_check["value"] == pytest.approx(45_000.0)
    assert exposure_check["threshold"] == pytest.approx(50_000.0)
    assert exposure_check["ratio"] == pytest.approx(0.9, rel=1e-6)

    leverage_check = next(
        check for check in decision.checks if check["name"] == "policy.max_leverage"
    )
    assert leverage_check["status"] == "warn"
    assert leverage_check["value"] == pytest.approx(0.45, rel=1e-6)
    assert leverage_check["threshold"] == pytest.approx(0.5)
    assert leverage_check["ratio"] == pytest.approx(0.9, rel=1e-6)

    assert decision.metadata["projected_total_exposure"] == pytest.approx(45_000.0)
    assert decision.metadata["projected_leverage"] == pytest.approx(0.45, rel=1e-6)
    assert decision.metadata["max_total_exposure"] == pytest.approx(50_000.0)
    assert decision.metadata["estimated_risk"] == pytest.approx(150.0)


def test_risk_policy_requires_stop_loss_and_equity_budget() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.4"),
        max_leverage=Decimal("3.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        mandatory_stop_loss=True,
    )
    policy = RiskPolicy.from_config(config)

    state = {
        "equity": 0.0,
        "cash": 0.0,
        "current_price": 1.15,
        "open_positions": {},
        "current_daily_drawdown": 0.2,
    }

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=2_000.0,
        price=0.0,
        stop_loss_pct=0.0,
        portfolio_state=state,
    )

    assert not decision.approved
    assert decision.reason == "policy.equity"
    assert "policy.stop_loss" in decision.violations
    assert "policy.equity" in decision.violations
    assert "policy.max_risk_per_trade_pct" in decision.violations
    assert "policy.max_drawdown_pct" in decision.violations
    assert decision.metadata["resolved_price"] == pytest.approx(1.15)
    assert decision.metadata["risk_budget"] == pytest.approx(0.0)
    max_risk_check = next(
        check for check in decision.checks if check["name"] == "policy.max_risk_per_trade_pct"
    )
    assert max_risk_check["ratio"] is None

    exposure_check = next(
        check for check in decision.checks if check["name"] == "policy.max_total_exposure_pct"
    )
    assert exposure_check["ratio"] is None


def test_risk_policy_resolves_price_from_existing_position_value() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.05"),
        max_total_exposure_pct=Decimal("0.9"),
        max_leverage=Decimal("10"),
        max_drawdown_pct=Decimal("0.5"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = _state(
        {
            "EURUSD": {
                "quantity": 2_000.0,
                "current_value": 2_200.0,
            }
        }
    )

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1_000.0,
        price=0.0,
        stop_loss_pct=0.02,
        portfolio_state=state,
    )

    assert decision.approved
    assert decision.metadata["resolved_price"] == pytest.approx(1.1)
    assert decision.metadata["existing_position_notional"] == pytest.approx(2_200.0)
    assert decision.metadata["projected_notional"] == pytest.approx(3_300.0)


def test_risk_policy_falls_back_to_portfolio_price_when_missing_market_price() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.05"),
        max_total_exposure_pct=Decimal("0.9"),
        max_leverage=Decimal("10"),
        max_drawdown_pct=Decimal("0.5"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = dict(_state())
    state["current_price"] = 1.25

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1_000.0,
        price=0.0,
        stop_loss_pct=0.02,
        portfolio_state=state,
    )

    assert decision.approved
    assert decision.metadata["resolved_price"] == pytest.approx(1.25)
    assert decision.metadata["projected_total_exposure"] == pytest.approx(1_250.0)


def test_risk_policy_rejects_when_market_price_unavailable() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=1_000.0,
        price=0.0,
        stop_loss_pct=0.02,
        portfolio_state=_state(),
    )

    assert not decision.approved
    assert decision.reason == "policy.market_price"
    assert "policy.market_price" in decision.violations
    market_check = next(
        check for check in decision.checks if check["name"] == "policy.market_price"
    )
    assert market_check["status"] == "violation"
    assert decision.metadata["resolved_price"] == pytest.approx(0.0)


def test_risk_policy_enforces_minimum_position_size() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.4"),
        max_leverage=Decimal("3.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=100.0,
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=50.0,
        price=1.2,
        stop_loss_pct=0.02,
        portfolio_state=_state(),
    )

    assert not decision.approved
    assert decision.reason == "policy.min_position_size"
    assert "policy.min_position_size" in decision.violations
    assert any(check["name"] == "policy.min_position_size" for check in decision.checks)


def test_risk_policy_limit_snapshot_serialises_thresholds() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.01"),
        max_total_exposure_pct=Decimal("0.2"),
        max_leverage=Decimal("3"),
        max_drawdown_pct=Decimal("0.05"),
        min_position_size=10,
        max_position_size=1000,
    )
    policy = RiskPolicy.from_config(config)

    snapshot = policy.limit_snapshot()

    assert snapshot == {
        "max_total_exposure_pct": pytest.approx(0.2),
        "max_leverage": pytest.approx(3.0),
        "max_risk_per_trade_pct": pytest.approx(0.01),
        "min_position_size": pytest.approx(10.0),
        "max_position_size": pytest.approx(1000.0),
        "max_drawdown_pct": pytest.approx(0.05),
    }


def test_risk_policy_flags_leverage_warning_before_violation() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("1.0"),
        max_leverage=Decimal("1.2"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = {
        "equity": 1_000.0,
        "open_positions": {"EURUSD": {"quantity": 400.0, "last_price": 1.0}},
        "current_daily_drawdown": 0.02,
    }

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=560.0,
        price=1.0,
        stop_loss_pct=0.01,
        portfolio_state=state,
    )

    leverage_check = next(
        check for check in decision.checks if check["name"] == "policy.max_leverage"
    )
    assert leverage_check["status"] == "warn"
    assert leverage_check["threshold"] == pytest.approx(1.2)
    assert leverage_check["value"] >= leverage_check["threshold"] * 0.8
    assert decision.approved is True


def test_risk_policy_rejects_when_exceeding_max_position_size() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("5.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        max_position_size=Decimal("5_000"),
    )
    policy = RiskPolicy.from_config(config)

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=10_000.0,
        price=1.0,
        stop_loss_pct=0.01,
        portfolio_state=_state(),
    )

    assert decision.approved is False
    assert decision.reason == "policy.max_position_size"
    assert "policy.max_position_size" in decision.violations

    max_position_check = next(
        check for check in decision.checks if check["name"] == "policy.max_position_size"
    )
    assert max_position_check["status"] == "violation"
    assert max_position_check["value"] == pytest.approx(10_000.0)
    assert max_position_check["threshold"] == pytest.approx(5_000.0)

    assert decision.metadata["violations"] == ("policy.max_position_size",)


def test_risk_policy_derives_equity_from_cash_and_positions() -> None:
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.4"),
        max_leverage=Decimal("3.0"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)

    state = {
        "equity": None,
        "cash": 5_000.0,
        "open_positions": {
            "EURUSD": {
                "quantity": 1_000.0,
                "last_price": 1.1,
            }
        },
        "current_daily_drawdown": 0.01,
    }

    decision = policy.evaluate(
        symbol="EURUSD",
        quantity=100.0,
        price=1.1,
        stop_loss_pct=0.01,
        portfolio_state=state,
    )

    assert decision.approved is True
    assert decision.violations == tuple()
    assert decision.metadata["equity"] == pytest.approx(6_100.0)
    assert decision.metadata["current_total_exposure"] == pytest.approx(1_100.0)
    assert decision.metadata["risk_budget"] == pytest.approx(122.0)
    total_exposure_check = next(
        check for check in decision.checks if check["name"] == "policy.max_total_exposure_pct"
    )
    assert total_exposure_check["status"] == "ok"
    assert total_exposure_check["ratio"] == pytest.approx(
        decision.metadata["projected_total_exposure"] / decision.metadata["max_total_exposure"]
    )
