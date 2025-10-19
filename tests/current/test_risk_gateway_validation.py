from __future__ import annotations

import asyncio
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
from src.risk.position_sizing import position_size
from src.risk.real_risk_manager import RealRiskConfig, RealRiskManager
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.risk_api import RISK_API_RUNBOOK
from src.trading.risk.policy_telemetry import RISK_POLICY_VIOLATION_RUNBOOK
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


class TrackingSizer:
    """Test helper that records position sizing invocations."""

    def __init__(self, recommended: Decimal = Decimal("5")) -> None:
        self.recommended = Decimal(recommended)
        self.calls: list[tuple[Decimal, Decimal, Decimal]] = []

    def __call__(
        self, balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal
    ) -> Decimal:
        self.calls.append((balance, risk_per_trade, stop_loss_pct))
        return self.recommended


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

    portfolio_monitor.portfolio["cash"] = 20000.0
    portfolio_monitor.portfolio["equity"] = 50000.0
    portfolio_monitor.portfolio["current_daily_drawdown"] = 0.0
    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.12

    result = await gateway.validate_trade_intent(Intent("EURUSD", Decimal("1")), state)

    assert result is None
    last_decision = gateway.get_last_decision()
    assert last_decision is not None
    assert last_decision.get("reason") == "max_drawdown_exceeded"
    reference = last_decision.get("risk_reference")
    assert isinstance(reference, Mapping)
    assert reference["risk_api_runbook"] == RISK_API_RUNBOOK
    summary = reference.get("risk_config_summary")
    assert isinstance(summary, Mapping)
    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.02)
    assert gateway.get_last_policy_decision() is None
    incident = gateway.get_last_guardrail_incident()
    assert incident is not None
    assert incident.severity == "violation"
    assert gateway.telemetry["guardrail_violations"] == 1
    assert gateway.telemetry["guardrail_near_misses"] == 0


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_on_synthetic_invariant_breach(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
    )

    state = portfolio_monitor.get_state()
    state["synthetic_invariant_breach"] = True

    result = await gateway.validate_trade_intent(Intent("EURUSD", Decimal("1")), state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "synthetic_invariant_breach"
    checks = decision.get("checks")
    assert isinstance(checks, list)
    guardrail = next(
        (entry for entry in checks if entry.get("name") == "risk.synthetic_invariant_posture"),
        None,
    )
    assert guardrail is not None
    assert guardrail.get("status") == "violation"
    metadata = guardrail.get("metadata")
    assert isinstance(metadata, Mapping)
    assert metadata.get("indicator") == "synthetic_invariant_breach"
    incident = gateway.get_last_guardrail_incident()
    assert incident is not None
    assert incident.severity == "violation"
    assert "risk.synthetic_invariant_posture" in incident.metadata.get("violations", [])
    assert gateway.telemetry["guardrail_violations"] >= 1


@pytest.mark.asyncio()
async def test_risk_gateway_detects_synthetic_invariant_in_metadata(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
    )

    state = portfolio_monitor.get_state()
    state["metadata"] = {
        "guardrails": {
            "syntheticInvariant": {
                "status": "BREACHED",
                "violations": ["shadow_notional"],
            }
        }
    }

    result = await gateway.validate_trade_intent(Intent("EURUSD", Decimal("1")), state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "synthetic_invariant_breach"
    guardrail_entry = next(
        (
            entry
            for entry in decision.get("checks", [])
            if entry.get("name") == "risk.synthetic_invariant_posture"
        ),
        None,
    )
    assert guardrail_entry is not None
    assert guardrail_entry.get("status") == "violation"
    metadata = guardrail_entry.get("metadata")
    assert isinstance(metadata, Mapping)
    assert metadata.get("indicator", "").startswith("metadata")
    details = metadata.get("details", {})
    assert isinstance(details, Mapping)
    assert details.get("guardrails") is not None

    incident = gateway.get_last_guardrail_incident()
    assert incident is not None
    assert incident.severity == "violation"
    assert "risk.synthetic_invariant_posture" in incident.metadata.get("violations", [])


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
    last_decision = gateway.get_last_decision()
    assert last_decision is not None
    assert last_decision.get("status") == "approved"
    reference = last_decision.get("risk_reference")
    assert isinstance(reference, Mapping)
    assert reference["risk_api_runbook"] == RISK_API_RUNBOOK
    policy_decision = gateway.get_last_policy_decision()
    assert policy_decision is not None
    assert policy_decision.approved is True
    snapshot = gateway.get_last_policy_snapshot()
    assert snapshot is not None
    assert snapshot.approved is True


@pytest.mark.asyncio()
async def test_risk_gateway_liquidity_probe_uses_portfolio_price(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    liquidity_prober = DummyLiquidityProber(score=0.75)
    policy = RiskPolicy.from_config(RiskConfig(min_position_size=1))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        liquidity_prober=liquidity_prober,
        liquidity_probe_threshold=1.0,
        min_liquidity_confidence=0.2,
        risk_policy=policy,
    )

    intent = {
        "symbol": "EURUSD",
        "quantity": Decimal("2"),
        "metadata": {},
    }

    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.0
    state["current_price"] = 1.2345

    validated = await gateway.validate_trade_intent(intent, state)

    assert validated is intent
    assert liquidity_prober.last_probe is not None
    symbol, price_levels, side = liquidity_prober.last_probe
    assert symbol == "EURUSD"
    assert side == "buy"
    assert len(price_levels) == 5
    assert any(level != 0.0 for level in price_levels)
    assert price_levels[2] == pytest.approx(1.2345, rel=1e-6)


@pytest.mark.asyncio()
async def test_risk_gateway_records_guardrail_near_miss(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    risk_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.1"),
        max_total_exposure_pct=Decimal("1.0"),
        max_leverage=Decimal("10.0"),
        max_drawdown_pct=Decimal("0.9"),
        min_position_size=1,
        mandatory_stop_loss=False,
    )

    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        max_daily_drawdown=0.95,
        risk_config=risk_config,
        risk_policy=None,
    )
    gateway._execution_config = ExecutionConfig(
        limits=ExecutionRiskLimits(
            max_slippage_bps=50.0,
            max_total_cost_bps=50.0,
            max_notional_pct_of_equity=1.0,
        )
    )

    portfolio_monitor.portfolio["cash"] = 10_000.0
    portfolio_monitor.portfolio["equity"] = 10_000.0
    portfolio_monitor.portfolio["current_daily_drawdown"] = 0.0

    intent = Intent("EURUSD", Decimal("7600"))
    intent.price = Decimal("1.0")
    intent.confidence = 0.95
    intent.metadata["stop_loss_pct"] = 0.01

    state = portfolio_monitor.get_state()
    state["equity"] = 10_000.0
    state["current_daily_drawdown"] = 0.0

    validated = await gateway.validate_trade_intent(intent, state)

    assert validated is intent
    incident = gateway.get_last_guardrail_incident()
    assert incident is not None
    assert incident.severity == "near_miss"
    assert gateway.telemetry["guardrail_near_misses"] == 1
    assert gateway.telemetry["guardrail_violations"] == 0


@pytest.mark.asyncio()
async def test_risk_gateway_converts_stop_loss_pips_for_position_sizer(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    sizer = TrackingSizer(recommended=Decimal("5"))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=sizer,
        portfolio_monitor=portfolio_monitor,
        risk_per_trade=Decimal("0.01"),
        stop_loss_floor=0.001,
        risk_policy=None,
    )

    portfolio_monitor.portfolio["equity"] = 50_000.0
    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.0
    state["pip_value"] = 0.0001

    intent = Intent("EURUSD", Decimal("5"))
    intent.price = Decimal("1.2000")
    intent.metadata["stop_loss_pips"] = 25

    validated = await gateway.validate_trade_intent(intent, state)

    assert validated is intent
    assert sizer.calls, "Expected position sizer to be invoked"
    _, _, stop_loss_arg = sizer.calls[0]
    expected_stop_loss = (25 * state["pip_value"]) / float(intent.price)
    assert float(stop_loss_arg) == pytest.approx(expected_stop_loss, rel=1e-6)

    assessment = intent.metadata.get("risk_assessment", {})
    checks = assessment.get("checks", [])
    assert any(check.get("name") == "position_sizer" for check in checks)


@pytest.mark.asyncio()
async def test_risk_gateway_stop_loss_pips_respects_floor(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    sizer = TrackingSizer(recommended=Decimal("5"))
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=sizer,
        portfolio_monitor=portfolio_monitor,
        risk_per_trade=Decimal("0.01"),
        stop_loss_floor=0.003,
        risk_policy=None,
    )

    portfolio_monitor.portfolio["equity"] = 25_000.0
    state = portfolio_monitor.get_state()
    state["current_daily_drawdown"] = 0.0
    state["pip_value"] = 0.00001

    intent = Intent("EURUSD", Decimal("5"))
    intent.price = Decimal("1.5000")
    intent.metadata["stop_loss_pips"] = 1

    validated = await gateway.validate_trade_intent(intent, state)

    assert validated is intent
    assert sizer.calls, "Expected position sizer to be invoked"
    _, _, stop_loss_arg = sizer.calls[0]
    assert float(stop_loss_arg) == pytest.approx(0.003, rel=1e-6)

@pytest.mark.asyncio()
async def test_risk_gateway_apply_risk_config_refreshes_limits(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    base_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.50"),
    )
    base_policy = RiskPolicy.from_config(base_config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=base_policy,
        risk_config=base_config,
    )

    updated_config = base_config.copy(
        update={
            "max_risk_per_trade_pct": Decimal("0.03"),
            "max_total_exposure_pct": Decimal("0.70"),
            "max_drawdown_pct": Decimal("0.15"),
            "mandatory_stop_loss": False,
            "research_mode": True,
        }
    )
    updated_policy = RiskPolicy.from_config(updated_config)

    gateway.apply_risk_config(updated_config, risk_policy=updated_policy)

    limits_payload = gateway.get_risk_limits()
    assert limits_payload["limits"]["max_risk_per_trade_pct"] == pytest.approx(0.03)
    summary = limits_payload["risk_config_summary"]
    assert summary["max_total_exposure_pct"] == pytest.approx(0.70)
    assert summary["mandatory_stop_loss"] is False
    assert summary["research_mode"] is True
    assert gateway.risk_policy is updated_policy
    assert gateway.risk_per_trade == Decimal("0.03")
    assert gateway.max_daily_drawdown == pytest.approx(0.15)


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
async def test_risk_gateway_policy_violation_emits_telemetry(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    restrictive_policy = RiskPolicy.from_config(
        RiskConfig(min_position_size=10_000, max_total_exposure_pct=Decimal("0.05"))
    )
    event_bus = EventBus()
    await event_bus.start()

    events: list[Any] = []
    received = asyncio.Event()

    async def _on_violation(event: Any) -> None:
        events.append(event)
        received.set()

    handle = event_bus.subscribe("telemetry.risk.policy_violation", _on_violation)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=restrictive_policy,
        event_bus=event_bus,
    )

    state = portfolio_monitor.get_state()
    state["open_positions"] = {}
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0

    intent = Intent("EURUSD", Decimal("10"))

    try:
        result = await gateway.validate_trade_intent(intent, state)
        assert result is None

        await asyncio.wait_for(received.wait(), timeout=1.0)
    finally:
        event_bus.unsubscribe(handle)
        await event_bus.stop()

    assert events, "expected policy violation telemetry"
    payload = events[-1].payload
    assert payload["runbook"] == RISK_POLICY_VIOLATION_RUNBOOK
    snapshot = payload.get("snapshot", {})
    assert snapshot.get("approved") is False
    assert payload.get("severity") == "critical"


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


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_when_confidence_notional_exceeded(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.50"),
        max_leverage=Decimal("5.0"),
        min_position_size=1,
    )
    policy = RiskPolicy.from_config(config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        risk_config=config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {}

    intent = Intent(
        "EURUSD",
        Decimal("20000"),
        price=Decimal("1.0"),
        confidence=0.2,
    )

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "confidence_notional_limit"
    checks = decision.get("checks", [])
    assert any(
        entry.get("name") == "risk.confidence_notional_limit"
        and entry.get("status") == "violation"
        for entry in checks
    )


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_when_leverage_limit_exceeded(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_total_exposure_pct=Decimal("1.0"),
        max_leverage=Decimal("2.0"),
        min_position_size=1,
    )
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=None,
        risk_config=config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {
        "EURUSD": {
            "quantity": 150_000,
            "avg_price": 1.0,
        }
    }

    intent = Intent(
        "GBPUSD",
        Decimal("100000"),
        price=Decimal("1.0"),
        confidence=1.0,
    )

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "leverage_limit"
    checks = decision.get("checks", [])
    assert any(
        entry.get("name") == "risk.leverage_ratio"
        and entry.get("status") == "violation"
        for entry in checks
    )


@pytest.mark.asyncio()
async def test_risk_gateway_rejects_when_sector_limit_exceeded(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.60"),
        max_leverage=Decimal("4.0"),
        min_position_size=1,
        instrument_sector_map={"EURUSD": "FX"},
        sector_exposure_limits={"FX": Decimal("0.10")},
    )
    policy = RiskPolicy.from_config(config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        risk_config=config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {
        "EURUSD": {
            "quantity": 9000,
            "avg_price": 1.0,
        }
    }

    intent = Intent(
        "EURUSD",
        Decimal("2000"),
        price=Decimal("1.0"),
        confidence=0.9,
    )

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "sector_exposure_limit"
    checks = decision.get("checks", [])
    assert any(
        entry.get("name") == "risk.sector_limit.FX"
        and entry.get("status") == "violation"
        for entry in checks
    )


@pytest.mark.asyncio()
async def test_risk_gateway_sector_limits_use_intent_metadata(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.60"),
        max_leverage=Decimal("4.0"),
        min_position_size=1,
        sector_exposure_limits={"FX": Decimal("0.10")},
    )
    policy = RiskPolicy.from_config(config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        risk_config=config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {
        "EURUSD": {
            "quantity": 9000,
            "avg_price": 1.0,
            "metadata": {"sector": "fx"},
        }
    }

    intent = Intent(
        "EURUSD",
        Decimal("2000"),
        price=Decimal("1.0"),
        confidence=0.9,
        metadata={"sector": "FX"},
    )

    result = await gateway.validate_trade_intent(intent, state)

    assert result is None
    decision = gateway.get_last_decision()
    assert decision is not None
    assert decision.get("reason") == "sector_exposure_limit"
    checks = decision.get("checks", [])
    assert any(
        entry.get("name") == "risk.sector_limit.FX"
        and entry.get("status") == "violation"
        for entry in checks
    )


@pytest.mark.asyncio()
async def test_risk_gateway_marks_unmapped_sector(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_total_exposure_pct=Decimal("0.60"),
        max_leverage=Decimal("4.0"),
        min_position_size=1,
        sector_exposure_limits={"FX": Decimal("0.50")},
    )
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=None,
        portfolio_monitor=portfolio_monitor,
        risk_policy=None,
        risk_config=config,
    )

    state = portfolio_monitor.get_state()
    state["equity"] = 100_000.0
    state["current_daily_drawdown"] = 0.0
    state["open_positions"] = {}

    intent = Intent(
        "EURUSD",
        Decimal("1000"),
        price=Decimal("1.0"),
        confidence=0.9,
    )

    result = await gateway.validate_trade_intent(intent, state)

    assert result is not None
    decision = gateway.get_last_decision()
    assert decision is not None
    checks = decision.get("checks", [])
    assert any(
        entry.get("name") == "risk.sector_limit.unmapped"
        and entry.get("status") == "info"
        for entry in checks
    )


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
    reference = limits.get("risk_reference")
    assert isinstance(reference, dict)
    assert reference.get("risk_api_runbook") == RISK_API_RUNBOOK
    reference_summary = reference.get("risk_config_summary")
    assert isinstance(reference_summary, dict)
    assert reference_summary["mandatory_stop_loss"] is True
    config_payload = reference.get("risk_config")
    assert isinstance(config_payload, dict)
    assert config_payload.get("mandatory_stop_loss") is True


@pytest.mark.asyncio()
async def test_risk_gateway_decision_includes_risk_reference(
    portfolio_monitor: PortfolioMonitor,
) -> None:
    registry = DummyStrategyRegistry(active=True)
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        mandatory_stop_loss=True,
        min_position_size=Decimal("1"),
        research_mode=True,
    )
    policy = RiskPolicy.from_config(config)
    gateway = RiskGateway(
        strategy_registry=registry,
        position_sizer=position_size,
        portfolio_monitor=portfolio_monitor,
        risk_policy=policy,
        risk_config=config,
    )

    intent = Intent("EURUSD", Decimal("1"), confidence=0.9)
    intent.metadata["stop_loss_pct"] = 0.01

    result = await gateway.validate_trade_intent(intent, None)

    assert result is not None
    decision = gateway.get_last_decision()
    assert decision is not None
    reference = decision.get("risk_reference")
    assert isinstance(reference, dict)
    assert reference.get("risk_api_runbook") == RISK_API_RUNBOOK
    summary = reference.get("risk_config_summary")
    assert isinstance(summary, dict)
    assert summary["max_total_exposure_pct"] == pytest.approx(float(config.max_total_exposure_pct))
    limits_snapshot = reference.get("limits")
    assert isinstance(limits_snapshot, dict)
    assert limits_snapshot["max_open_positions"] == gateway.max_open_positions

    risk_assessment = intent.metadata.get("risk_assessment", {})
    embedded_reference = risk_assessment.get("risk_reference")
    assert isinstance(embedded_reference, dict)
    assert embedded_reference.get("risk_api_runbook") == RISK_API_RUNBOOK
