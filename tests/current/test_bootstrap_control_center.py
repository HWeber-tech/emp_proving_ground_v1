from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.core.event_bus import EventBus
from src.data_foundation import HistoricalReplayConnector, MarketDataFabric
from src.operations.bootstrap_control_center import BootstrapControlCenter
from src.orchestration.bootstrap_stack import BootstrapSensoryPipeline, BootstrapTradingStack
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
            {
                "timestamp": now,
                "open": 1.1055,
                "high": 1.1080,
                "low": 1.1045,
                "close": 1.1075,
                "volume": 2400,
                "volatility": 0.0005,
                "spread": 0.00005,
                "depth": 6400,
                "order_imbalance": 0.3,
                "macro_bias": 0.45,
                "data_quality": 0.94,
            },
        ]
    }
    return HistoricalReplayConnector(bars)


class DummyStrategyRegistry:
    def get_strategy(self, strategy_id: str) -> Mapping[str, Any] | None:
        return {"status": "active", "strategy_id": strategy_id}


class DummyChampion:
    def __init__(self) -> None:
        self.genome_id = "core-evo-00001"
        self.fitness = 0.87

    def as_payload(self) -> Mapping[str, Any]:
        return {"genome_id": self.genome_id, "fitness": self.fitness}


class DummyEvolutionOrchestrator:
    def __init__(self) -> None:
        self.telemetry = {"total_generations": 3, "champion": {"genome_id": "core-evo-00001"}}
        self.population_statistics = {"population_size": 12, "best_fitness": 0.87}
        self.champion = DummyChampion()


@pytest.mark.asyncio()
async def test_control_center_compiles_telemetry_report() -> None:
    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    fusion_engine = ContextualFusionEngine()
    pipeline = BootstrapSensoryPipeline(fabric, fusion_engine)

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
        min_intent_confidence=0.0,
    )

    orchestrator = DummyEvolutionOrchestrator()

    control_center = BootstrapControlCenter(
        pipeline=pipeline,
        trading_manager=trading_manager,
        execution_adapter=execution_adapter,
        liquidity_prober=liquidity_prober,
        evolution_orchestrator=orchestrator,
    )

    stack = BootstrapTradingStack(
        pipeline,
        trading_manager,
        strategy_id="bootstrap-alpha",
        buy_threshold=0.05,
        sell_threshold=0.05,
        requested_quantity=Decimal("1.5"),
        stop_loss_pct=0.01,
        liquidity_prober=liquidity_prober,
        control_center=control_center,
    )

    await stack.evaluate_tick("EURUSD")  # warm-up
    await stack.evaluate_tick("EURUSD")

    report = control_center.generate_report()

    assert report["portfolio"]["equity"] > 0
    assert report["risk"]["limits"]["max_open_positions"] == 5
    assert report["performance"]["fills"] >= 1
    assert report["decisions"]["recent"]
    assert report["intelligence"]["narrative"] in {"BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"}
    assert report["liquidity"]["summary"].get("evaluated_levels") is not None
    assert report["evolution"]["champion"]["genome_id"] == orchestrator.champion.genome_id
    assert report["evolution"]["population"]["population_size"] == 12
    assert report["vision_alignment"]["summary"]["status"] in {"ready", "progressing", "gap"}
    assert report["vision_alignment"]["layers"]

    overview = control_center.overview()
    assert overview["equity"] == report["performance"]["equity"]
    assert overview["last_decision"] is not None
    assert overview["evolution"]["generations"] == orchestrator.telemetry["total_generations"]
    assert overview["vision_alignment"]["status"] in {"ready", "progressing", "gap"}
