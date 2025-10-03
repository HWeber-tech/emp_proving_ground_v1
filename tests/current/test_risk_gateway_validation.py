from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.data_foundation.config.execution_config import (
    ExecutionConfig,
    ExecutionRiskLimits,
)
from src.core.base import MarketData
from src.core.event_bus import EventBus
from src.core.risk.position_sizing import position_size
from src.risk.real_risk_manager import RealRiskConfig, RealRiskManager
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.risk_api import RISK_API_RUNBOOK
from src.trading.risk.risk_gateway import RiskGateway
from src.trading.risk.risk_policy import RiskPolicy


class DummyStrategyRegistry:
    def __init__(self, *, active: bool = True) -> None:
        self._active = active

    def get_strategy(
        self, strategy_id: str
    ) -> Mapping[str, Any] | None:  # pragma: no cover - exercised in tests
        return {"status": "active" if self._active else "inactive"}


class DummyLiquidityProber:
    def __init__(self, score: float = 0.85) -> None:
        self.score = score
        self.last_probe: tuple[str, list[float], str] | None = None

    async def probe_liquidity(self, symbol: str, price_levels: list[float], side: str):
        self.last_probe = (symbol, price_levels, side)
        return {price: 2.0 for price in price_levels}

    def calculate_liquidity_confidence_score(self, probe_results, intended_volume: float) -> float:
        return self.score

    def get_probe_summary(self, probe_results) -> Mapping[str, Any]:
        return {
            "total_levels": len(probe_results),
            "total_liquidity": sum(probe_results.values()),
        }


@dataclass
class Intent:
    symbol: str
    quantity: Decimal
    side: str = "BUY"
    price: Decimal = Decimal("1.1000")
    confidence: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture()
def portfolio_monitor() -> PortfolioMonitor:
    event_bus = EventBus()
    monitor = PortfolioMonitor(event_bus, InMemoryRedis())
    monitor.portfolio["cash"] = 1000.0
    monitor.portfolio["daily_pnl"] = 0.0
    monitor.portfolio["open_positions"] = {}
    return monitor


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_when_drawdown_exceeded(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        max_daily_drawdown=0.05,
        risk_policy=policy,
    )

    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.12

    result = await gateway.validate_trade_intent(Intent("EURUSD", Decimal("1")), state)

    assert result is None
    last_decision = gateway.get_last_decision()
    assert last_decision is not None
    assert last_decision.get("reason") == "max_drawdown_exceeded"
    assert gateway.get_last_policy_decision() is None


@pytest.mark.asyncio()
async def test_risk_gateway_clips_position_and_adds_liquidity_metadata(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    liquidity_prober = DummyLiquidityProber(score=0.9)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=position_size,
        portfolio_monitor=portfolio_monitor,
        risk_per_trade=Decimal("0.001"),
        max_open_positions=5,
        liquidity_prober=liquidity_prober,
        liquidity_probe_threshold=1.0,
        min_liquidity_confidence=0.2,
        risk_policy=policy,
    )

    intent = Intent("GBPUSD", Decimal("5"))
    intent.metadata["stop_loss_pct"] = 0.5

    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.0

    validated = await gateway.validate_trade_intent(intent, state)

    assert validated is intent
    assert liquidity_prober.last_probe is not None
    # Risk gateway should clip down to recommended size from position sizing
    assert intent.quantity == Decimal("2")
    assessment = intent.metadata.get("risk_assessment", {})
    assert assessment.get("liquidity_confidence") == pytest.approx(0.9)
    assert gateway.get_last_decision().get("status") == "approved"
    policy_decision = gateway.get_last_policy_decision()
    assert policy_decision is not None
    assert policy_decision.approved is True
    snapshot = gateway.get_last_policy_snapshot()
    assert snapshot is not None
    assert snapshot.approved is True


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_on_insufficient_liquidity(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    liquidity_prober = DepthAwareLiquidityProber(max_history=4)

    now = datetime.now(timezone.utc)
    for depth in (40.0, 35.0):
        liquidity_prober.record_snapshot(
            "EURUSD",
            MarketData(
                symbol="EURUSD",
                timestamp=now,
                close=1.0998,
                volume=150.0,
                depth=depth,
                spread=0.0003,
                order_imbalance=-0.5,
                data_quality=0.55,
            ),
        )
        now += timedelta(seconds=1)

    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        liquidity_prober=liquidity_prober,
        liquidity_probe_threshold=10.0,
        min_liquidity_confidence=0.6,
        risk_policy=RiskPolicy.from_config(RiskConfig(min_position_size=1)),
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0

    intent = Intent("EURUSD", Decimal("1000"), confidence=0.95)

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "insufficient_liquidity"
    policy_decision = gateway.get_last_policy_decision()
    assert policy_decision is not None
    assert policy_decision.metadata["symbol"] == "EURUSD"


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_on_policy_violation(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    restrictive_policy = RiskPolicy.from_config(
        RiskConfig(min_position_size=10_000, max_total_exposure_pct=Decimal("0.05"))
    )
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=restrictive_policy,
    )

    state = portfolio_monitor.get_state()
    state["open_positions"] = {}
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0

    intent = Intent("EURUSD", Decimal("10"))
    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "policy.min_position_size"
    policy_decision = gateway.get_last_policy_decision()
    assert policy_decision is not None
    assert "policy.min_position_size" in policy_decision.violations
    snapshot = gateway.get_last_policy_snapshot()
    assert snapshot is not None
    assert "policy.min_position_size" in snapshot.violations


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_when_portfolio_risk_exceeded(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    portfolio_risk_manager = RealRiskManager(
        RealRiskConfig(
            max_position_risk=0.001,
            max_total_exposure=0.001,
            max_drawdown=0.25,
            max_leverage=10.0,
            equity=100_000.0,
        )
    )

    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=None,
        portfolio_risk_manager=portfolio_risk_manager,
        risk_per_trade=Decimal("0.001"),
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {
        "EURUSD": {
            "quantity": 40000,
            "avg_price": 1.0,
            "stop_loss_pct": 0.002,
        }
    }

    intent = Intent("GBPUSD", Decimal("10000"))
    intent.metadata["stop_loss_pct"] = 0.008

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "portfolio_risk_breach"
    risk_summary = decision.get("risk_manager")
    assert isinstance(risk_summary, Mapping)
    assert risk_summary.get("risk_score") and risk_summary["risk_score"] > 1.0

@pytest.mark.asyncio()
async def test_risk_gateway_rejects_on_execution_risk(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    execution_config = ExecutionConfig(
        limits=ExecutionRiskLimits(
            max_slippage_bps=10.0,
            max_total_cost_bps=15.0,
            max_notional_pct_of_equity=0.5,
        )
    )

    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        execution_config=execution_config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0

    intent = Intent("USDJPY", Decimal("100"), price=Decimal("100"))
    intent.metadata["microstructure"] = {
        "spread": 0.8,
        "liquidity_imbalance": 0.4,
        "price_volatility": 0.3,
    }

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "execution_risk"
    checks = decision.get("checks", [])
    assert any(
        check.get("name") == "execution.slippage_bps"
        and check.get("status") == "violation"
        for check in checks
    )
    exec_meta = intent.metadata.get("execution_risk", {})
    assert exec_meta.get("slippage_bps", 0.0) > execution_config.limits.max_slippage_bps


def test_risk_gateway_limits_include_risk_api_summary(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        mandatory_stop_loss=True,
    )
    policy = RiskPolicy.from_config(config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        risk_config=config,
    )

    limits = gateway.get_risk_limits()

    summary = limits.get("risk_config_summary")
    assert isinstance(summary, dict)
    assert summary["max_risk_per_trade_pct"] == pytest.approx(float(config.max_risk_per_trade_pct))
    assert summary["mandatory_stop_loss"] is True
    assert limits.get("runbook") == RISK_API_RUNBOOK
