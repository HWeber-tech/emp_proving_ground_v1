from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.core.event_bus import EventBus
from src.data_foundation import HistoricalReplayConnector, MarketDataFabric
from src.orchestration.bootstrap_stack import (
    BootstrapSensoryPipeline,
    BootstrapTradingStack,
)
from src.orchestration.enhanced_intelligence_engine import ContextualFusionEngine
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.trading_manager import TradingManager


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
    assert pipeline.history["EURUSD"]


@pytest.mark.asyncio()
async def test_bootstrap_trading_stack_executes_trade() -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    event_bus = EventBus()
    portfolio_monitor = PortfolioMonitor(event_bus, InMemoryRedis())
    execution_adapter = ImmediateFillExecutionAdapter(portfolio_monitor)
    liquidity_prober = DepthAwareLiquidityProber()
    trading_manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=DummyStrategyRegistry(),
        execution_engine=execution_adapter,
        initial_equity=100000.0,
        risk_per_trade=0.02,
        max_open_positions=5,
        max_daily_drawdown=0.1,
        redis_client=InMemoryRedis(),
        liquidity_prober=liquidity_prober,
    )

    stack = BootstrapTradingStack(
        pipeline,
        trading_manager,
        strategy_id="bootstrap-alpha",
        buy_threshold=0.15,
        sell_threshold=0.15,
        requested_quantity=Decimal("2"),
        stop_loss_pct=0.01,
        liquidity_prober=liquidity_prober,
    )

    await stack.evaluate_tick("EURUSD")  # warm-up tick
    result = await stack.evaluate_tick("EURUSD")
    assert result["decision"] is not None
    assert result["decision"]["status"] == "approved"
    assert execution_adapter.fills
    position_state = portfolio_monitor.get_state()["open_positions"].get("EURUSD")
    assert position_state is not None
    assert position_state["quantity"] > 0
    assert result["liquidity_summary"] is not None
