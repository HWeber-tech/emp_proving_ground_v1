from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.core.event_bus import EventBus
from src.data_foundation import HistoricalReplayConnector, MarketDataFabric
from src.operations import bootstrap_control_center as bootstrap_module
from src.operations.bootstrap_control_center import BootstrapControlCenter
from src.orchestration.bootstrap_stack import BootstrapSensoryPipeline, BootstrapTradingStack
from src.orchestration.enhanced_intelligence_engine import ContextualFusionEngine
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.risk_api import RISK_API_RUNBOOK
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
        min_intent_confidence=0.0,
        risk_config=risk_config,
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
    assert report["risk"]["snapshot"]["status"] in {"ok", "warn", "alert"}
    assert report["risk"]["policy"] is not None
    assert report["risk"]["policy"]["snapshot"]["symbol"] == "EURUSD"
    assert report["risk"]["runbook"].endswith("risk_api_contract.md")
    metadata = report["risk"].get("metadata")
    assert metadata is not None
    assert metadata.get("runbook", "").endswith("risk_api_contract.md")
    reference = report["risk"].get("risk_reference")
    assert reference is not None
    assert reference["risk_api_runbook"].endswith("risk_api_contract.md")
    summary = reference.get("risk_config_summary")
    assert summary is not None
    assert summary["max_risk_per_trade_pct"] == pytest.approx(float(risk_config.max_risk_per_trade_pct))
    interface = report["risk"].get("interface")
    assert interface is not None
    assert interface.get("summary", {}).get("runbook") == RISK_API_RUNBOOK
    assert report["performance"]["fills"] >= 1
    assert "roi" in report["performance"]
    assert report["performance"]["roi"]["snapshot"]["status"] in {"ahead", "tracking", "at_risk"}
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
    assert overview["risk_posture"]["status"] in {"ok", "warn", "alert"}
    assert overview["risk_policy"]["symbol"] == "EURUSD"
    interface_overview = overview.get("risk_interface")
    assert interface_overview is not None
    assert interface_overview.get("summary", {}).get("runbook") == RISK_API_RUNBOOK
    overview_reference = overview.get("risk_reference")
    assert overview_reference is not None
    assert overview_reference["risk_api_runbook"].endswith("risk_api_contract.md")
    assert overview.get("risk_runbook", "").endswith("risk_api_contract.md")
    overview_metadata = overview.get("risk_metadata")
    assert overview_metadata is not None
    assert overview_metadata.get("max_total_exposure_pct") == pytest.approx(
        float(risk_config.max_total_exposure_pct)
    )
    assert overview["roi_posture"]["status"] in {"ahead", "tracking", "at_risk"}
    assert overview["evolution"]["generations"] == orchestrator.telemetry["total_generations"]
    assert overview["vision_alignment"]["status"] in {"ready", "progressing", "gap"}


def test_trading_manager_method_failures_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingManager:
        def method(self) -> None:
            raise RuntimeError("boom")

    caplog.set_level(logging.WARNING, logger=bootstrap_module.__name__)

    manager = ExplodingManager()
    result = bootstrap_module._call_trading_manager_method(manager, "method")

    assert result is None
    assert "Trading manager method 'method' failed" in caplog.text


def test_formatter_failures_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    def explode_formatter(_: object) -> str:
        raise ValueError("bad format")

    caplog.set_level(logging.WARNING, logger=bootstrap_module.__name__)

    result = bootstrap_module._format_optional_markdown(explode_formatter, object())

    assert result is None
    assert "Formatter" in caplog.text and "bad format" in caplog.text
