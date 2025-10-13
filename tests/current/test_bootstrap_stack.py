from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.core.event_bus import EventBus
from src.data_foundation import HistoricalReplayConnector, MarketDataFabric
from src.orchestration.bootstrap_stack import (
    BootstrapSensoryPipeline,
    BootstrapTradingStack,
)
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.trading_manager import TradingManager
from src.understanding.decision_diary import DecisionDiaryStore



def _build_replay_connector() -> HistoricalReplayConnector:
    now = datetime.now(timezone.utc)
    bars: Mapping[str, list[Mapping[str, Any]]] = {
        "EURUSD": [
            {
                "timestamp": now - timedelta(minutes=2),
                "open": 1.1000,
                "high": 1.1030,
                "low": 1.0990,
                "close": 1.1025,
                "volume": 1800,
                "volatility": 0.0004,
                "spread": 0.00005,
                "depth": 5500,
                "order_imbalance": 0.2,
                "macro_bias": 0.35,
                "data_quality": 0.9,
            },
            {
                "timestamp": now - timedelta(minutes=1),
                "open": 1.1025,
                "high": 1.1060,
                "low": 1.1020,
                "close": 1.1055,
                "volume": 2200,
                "volatility": 0.00045,
                "spread": 0.00005,
                "depth": 6000,
                "order_imbalance": 0.28,
                "macro_bias": 0.4,
                "data_quality": 0.92,
            },
        ]
    }
    return HistoricalReplayConnector(bars)


class DummyStrategyRegistry:
    def get_strategy(self, strategy_id: str) -> Mapping[str, Any] | None:
        return {"status": "active", "strategy_id": strategy_id}


class _StubRiskGateway:
    def get_last_decision(self) -> Mapping[str, Any]:
        return {"checks": []}


class StubTradingManager:
    def __init__(self) -> None:
        self.intents: list[Any] = []
        self.risk_gateway = _StubRiskGateway()

    async def on_trade_intent(self, intent: Any) -> None:
        self.intents.append(intent)


@pytest.mark.asyncio()
async def test_historical_replay_connector_iterates() -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})

    first = await fabric.fetch_latest("EURUSD", use_cache=False)
    second = await fabric.fetch_latest("EURUSD", use_cache=False)

    assert second.timestamp > first.timestamp
    assert second.close != first.close


@pytest.mark.asyncio()
async def test_bootstrap_pipeline_generates_snapshots() -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    snapshot = await pipeline.process_tick("EURUSD")
    assert snapshot.symbol == "EURUSD"
    assert isinstance(snapshot.synthesis.unified_score, float)
    assert snapshot.latencies is not None
    assert snapshot.latencies["ingest"] > 0.0
    assert snapshot.latencies["signal"] > 0.0
    assert pipeline.history["EURUSD"]
    audit = pipeline.audit_trail(limit=1)
    assert audit
    entry = audit[0]
    assert entry["symbol"] == "EURUSD"
    assert set(entry["dimensions"].keys()) == {"WHY", "HOW", "WHAT", "WHEN", "ANOMALY"}


@pytest.mark.asyncio()
async def test_bootstrap_trading_stack_executes_trade() -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    event_bus = EventBus()
    portfolio_monitor = PortfolioMonitor(event_bus, InMemoryRedis())
    execution_adapter = ImmediateFillExecutionAdapter(portfolio_monitor)
    liquidity_prober = DepthAwareLiquidityProber()
    risk_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    trading_manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=DummyStrategyRegistry(),
        execution_engine=execution_adapter,
        initial_equity=100000.0,
        risk_per_trade=None,
        max_open_positions=5,
        max_daily_drawdown=None,
        redis_client=InMemoryRedis(),
        liquidity_prober=liquidity_prober,
        risk_config=risk_config,
    )

    stack = BootstrapTradingStack(
        pipeline,
        trading_manager,
        strategy_id="bootstrap-alpha",
        buy_threshold=0.05,
        sell_threshold=0.05,
        requested_quantity=Decimal("2"),
        stop_loss_pct=0.01,
        liquidity_prober=liquidity_prober,
    )

    await stack.evaluate_tick("EURUSD")  # warm-up tick
    result = await stack.evaluate_tick("EURUSD")
    assert result["intent"] is not None
    assert result["decision"] is not None
    assert result["decision"].get("status") == "approved"
    assert execution_adapter.fills
    position_state = portfolio_monitor.get_state()["open_positions"].get("EURUSD")
    assert position_state is not None
    assert position_state["quantity"] > 0
    assert result["liquidity_summary"] is not None
    policy_snapshot = trading_manager.get_last_policy_snapshot()
    assert policy_snapshot is not None
    assert policy_snapshot.symbol == "EURUSD"
    intent_metadata = getattr(result["intent"], "metadata", {})
    assert "understanding_snapshot" in intent_metadata
    assert "intelligence_snapshot" not in intent_metadata

    observability = stack.describe_pipeline_observability()
    heartbeat = observability["heartbeat"]
    assert heartbeat["ticks"] >= 2
    latency = observability["latency"]
    assert latency["ingest"]["samples"] >= 2
    assert latency["ack"]["samples"] >= 1
    assert latency["ack"]["p50"] is not None
    decision_latency = observability["decision_latency"]
    assert decision_latency["status"] in {"pass", "warn"}
    assert decision_latency["baseline"]["p50_s"] > 0


@pytest.mark.asyncio()
async def test_bootstrap_pipeline_logs_listener_failure(caplog: pytest.LogCaptureFixture) -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    def failing_listener(_: Any) -> None:
        raise RuntimeError("boom")

    pipeline.register_listener(failing_listener)

    with caplog.at_level(logging.ERROR):
        snapshot = await pipeline.process_tick("EURUSD")

    assert snapshot.symbol == "EURUSD"
    messages = [record.message for record in caplog.records]
    assert any("Bootstrap sensory listener failed" in message for message in messages)


class ExplodingProber:
    def record_snapshot(self, symbol: str, market_data: Any) -> None:
        raise RuntimeError(f"{symbol} failure")


class ExplodingControlCenter:
    def record_tick(self, *, snapshot: Any, result: Mapping[str, Any]) -> None:
        raise RuntimeError("control center offline")


@pytest.mark.asyncio()
async def test_bootstrap_stack_logs_optional_callback_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    stack = BootstrapTradingStack(
        pipeline,
        StubTradingManager(),
        liquidity_prober=ExplodingProber(),
        control_center=ExplodingControlCenter(),
    )

    with caplog.at_level(logging.ERROR):
        await stack.evaluate_tick("EURUSD")

    messages = [record.message for record in caplog.records]
    assert any("Liquidity prober snapshot recording failed" in message for message in messages)
    assert any("Bootstrap control center notification failed" in message for message in messages)


@pytest.mark.asyncio()
async def test_decision_diary_records_trade_throttle_note(tmp_path: Any) -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    event_bus = EventBus()
    portfolio_monitor = PortfolioMonitor(event_bus, InMemoryRedis())
    execution_adapter = ImmediateFillExecutionAdapter(portfolio_monitor)
    liquidity_prober = DepthAwareLiquidityProber()
    risk_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
    )
    trading_manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=DummyStrategyRegistry(),
        execution_engine=execution_adapter,
        initial_equity=100000.0,
        redis_client=InMemoryRedis(),
        liquidity_prober=liquidity_prober,
        risk_config=risk_config,
        trade_throttle={"max_trades": 1, "window_seconds": 120.0},
    )

    diary_path = tmp_path / "decision_diary.json"
    diary_store = DecisionDiaryStore(diary_path, publish_on_record=False)

    stack = BootstrapTradingStack(
        pipeline,
        trading_manager,
        strategy_id="bootstrap-alpha",
        buy_threshold=-1.0,
        sell_threshold=0.05,
        requested_quantity=Decimal("1"),
        stop_loss_pct=0.01,
        liquidity_prober=liquidity_prober,
        diary_store=diary_store,
    )

    await stack.evaluate_tick("EURUSD")
    throttled_result = await stack.evaluate_tick("EURUSD")

    assert throttled_result["status"] == "throttled"

    entries = diary_store.entries()
    assert len(entries) >= 2
    throttled_entry = entries[-1]
    assert any("throttle" in note.lower() for note in throttled_entry.notes)

    throttle_snapshot = throttled_entry.outcomes.get("throttle")
    assert isinstance(throttle_snapshot, Mapping)
    assert throttle_snapshot.get("state") in {"rate_limited", "cooldown", "min_interval"}
