from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.core.event_bus import AsyncEventBus, Event, EventBus
from src.data_foundation import HistoricalReplayConnector, MarketDataFabric
from src.orchestration.bootstrap_stack import (
    BootstrapSensoryPipeline,
    BootstrapTradingStack,
)
from src.orchestration.compose import compose_validation_adapters
from src.orchestration.enhanced_understanding_engine import ContextualFusionEngine
from src.risk.risk_manager_impl import RiskManagerImpl
from src.trading.execution.execution_engine import ExecutionEngine
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.trading_manager import TradingManager
from tests.util.orchestration_stubs import (
    InMemoryStateStore,
    install_phase3_orchestrator,
)


@pytest.mark.asyncio
async def test_orchestration_risk_execution_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = compose_validation_adapters()
    orchestrator_module = install_phase3_orchestrator(monkeypatch)

    event_bus = EventBus()
    await event_bus.start()

    state_store = InMemoryStateStore()
    orchestrator = orchestrator_module.Phase3Orchestrator(
        state_store=state_store,
        event_bus=event_bus,
        adaptation_service=adapters["adaptation_service"],
    )
    risk_manager = RiskManagerImpl(initial_balance=75_000, risk_config=RiskConfig())
    execution = ExecutionEngine()

    recorded_events: list[tuple[str, dict[str, Any]]] = []

    async def _collector(event: Event) -> None:
        recorded_events.append((event.type, event.payload or {}))

    handle = event_bus.subscribe("trade.execution", _collector)

    position_size = 0.0

    try:
        assert await orchestrator.initialize() is True

        analysis = await orchestrator.run_full_analysis()
        predictive = analysis["systems"]["predictive"]
        signal = {
            "symbol": "EURUSD",
            "confidence": predictive["average_confidence"],
            "stop_loss_pct": 0.015,
        }

        assert await risk_manager.validate_position(
            {"symbol": "EURUSD", "size": 10_000, "entry_price": 1.2345, "stop_loss_pct": 0.015}
        )

        position_size = await risk_manager.calculate_position_size(signal)
        assert position_size > 0

        risk_manager.add_position("EURUSD", position_size, 1.2345, stop_loss_pct=0.015)

        order_id = await execution.send_order(
            "EURUSD",
            "BUY",
            position_size,
            price=1.2345,
        )
        execution.record_fill(order_id, position_size * 0.4, 1.2347)
        execution.record_fill(order_id, position_size * 0.6, 1.2351)

        risk_manager.update_position_value("EURUSD", 1.2351)

        reconciliation = execution.reconcile()
        filled_snapshot = next(
            entry for entry in reconciliation["filled_orders"] if entry["order_id"] == order_id
        )

        await event_bus.publish(
            Event(
                type="trade.execution",
                payload={
                    "order": filled_snapshot,
                    "positions": reconciliation["positions"],
                    "risk": risk_manager.get_position_risk("EURUSD"),
                },
            )
        )
        await asyncio.sleep(0.05)
    finally:
        event_bus.unsubscribe(handle)
        await orchestrator.stop()
        await event_bus.stop()

    assert recorded_events, "expected execution telemetry to be emitted"
    event_type, payload = recorded_events[-1]
    assert event_type == "trade.execution"
    assert payload["order"]["status"] == "FILLED"
    assert payload["risk"]["symbol"] == "EURUSD"
    assert payload["positions"]["EURUSD"]["quantity"] == pytest.approx(position_size)


def _build_replay_connector() -> HistoricalReplayConnector:
    now = datetime.now(timezone.utc)
    bars: Mapping[str, list[Mapping[str, Any]]] = {
        "EURUSD": [
            {
                "timestamp": now - timedelta(minutes=3),
                "open": 1.0990,
                "high": 1.1035,
                "low": 1.0980,
                "close": 1.1020,
                "volume": 1600,
                "volatility": 0.00035,
                "spread": 0.00004,
                "depth": 5200,
                "order_imbalance": 0.22,
                "macro_bias": 0.32,
                "data_quality": 0.9,
            },
            {
                "timestamp": now - timedelta(minutes=2),
                "open": 1.1020,
                "high": 1.1055,
                "low": 1.1015,
                "close": 1.1045,
                "volume": 2100,
                "volatility": 0.0004,
                "spread": 0.00005,
                "depth": 5800,
                "order_imbalance": 0.27,
                "macro_bias": 0.36,
                "data_quality": 0.92,
            },
            {
                "timestamp": now - timedelta(minutes=1),
                "open": 1.1045,
                "high": 1.1080,
                "low": 1.1035,
                "close": 1.1075,
                "volume": 2400,
                "volatility": 0.00042,
                "spread": 0.00005,
                "depth": 6000,
                "order_imbalance": 0.3,
                "macro_bias": 0.4,
                "data_quality": 0.94,
            },
        ]
    }
    return HistoricalReplayConnector(bars)


class _AlwaysActiveRegistry:
    def get_strategy(
        self, strategy_id: str
    ) -> Mapping[str, Any] | None:  # pragma: no cover - simple stub
        return {"strategy_id": strategy_id, "status": "active"}


class _TopicRecorder:
    def __init__(self, topics: tuple[str, ...]) -> None:
        self._topics = topics
        self.events: dict[str, list[Event]] = {topic: [] for topic in topics}
        self._signals = {topic: asyncio.Event() for topic in topics}

    async def handler(self, event: Event) -> None:
        if event.type not in self.events:
            return
        self.events[event.type].append(event)
        signal = self._signals.get(event.type)
        if signal is not None and not signal.is_set():
            signal.set()

    async def wait_for_all(self, timeout: float = 1.5) -> None:
        await asyncio.wait_for(
            asyncio.gather(*(signal.wait() for signal in self._signals.values())),
            timeout=timeout,
        )

    def snapshot_counts(self) -> dict[str, int]:
        return {topic: len(self.events.get(topic, [])) for topic in self._topics}


@pytest.mark.asyncio()
async def test_bootstrap_stack_event_bus_fallback_flow() -> None:
    topics = (
        "telemetry.risk.posture",
        "telemetry.risk.policy",
        "telemetry.operational.roi",
    )

    event_bus = AsyncEventBus()
    await event_bus.start()
    recorder = _TopicRecorder(topics)
    handles = [event_bus.subscribe(topic, recorder.handler) for topic in topics]

    connector = _build_replay_connector()
    fabric = MarketDataFabric({"replay": connector})
    pipeline = BootstrapSensoryPipeline(fabric, ContextualFusionEngine())

    portfolio_monitor = PortfolioMonitor(event_bus, InMemoryRedis())
    execution_adapter = ImmediateFillExecutionAdapter(portfolio_monitor)
    liquidity_prober = DepthAwareLiquidityProber()
    risk_config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        max_position_size=50_000,
    )

    trading_manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=_AlwaysActiveRegistry(),
        execution_engine=execution_adapter,
        initial_equity=100_000.0,
        redis_client=InMemoryRedis(),
        liquidity_prober=liquidity_prober,
        risk_config=risk_config,
        min_intent_confidence=0.05,
    )

    stack = BootstrapTradingStack(
        pipeline,
        trading_manager,
        strategy_id="bootstrap-integration",
        buy_threshold=0.05,
        sell_threshold=0.05,
        requested_quantity=Decimal("2"),
        stop_loss_pct=0.01,
        liquidity_prober=liquidity_prober,
    )

    try:
        await stack.evaluate_tick("EURUSD")
        first_result = await stack.evaluate_tick("EURUSD")
        assert first_result["status"] == "submitted"
        decision = first_result["decision"]
        assert isinstance(decision, Mapping)
        assert decision.get("status") == "approved"

        await recorder.wait_for_all()
        assert execution_adapter.fills

        baseline_counts = recorder.snapshot_counts()
        risk_snapshot = trading_manager.get_last_risk_snapshot()
        policy_snapshot = trading_manager.get_last_policy_snapshot()
        roi_snapshot = trading_manager.get_last_roi_snapshot()

        assert risk_snapshot is not None
        assert policy_snapshot is not None
        assert roi_snapshot is not None
        assert roi_snapshot.executed_trades >= 1

        first_fill_count = len(execution_adapter.fills)

        await event_bus.stop()

        fallback_result = await stack.evaluate_tick("EURUSD")
        assert fallback_result["status"] == "submitted"
        assert len(execution_adapter.fills) == first_fill_count + 1

        fallback_risk = trading_manager.get_last_risk_snapshot()
        fallback_policy = trading_manager.get_last_policy_snapshot()
        fallback_roi = trading_manager.get_last_roi_snapshot()

        assert fallback_risk is not None
        assert fallback_policy is not None
        assert fallback_roi is not None
        assert fallback_roi.executed_trades > roi_snapshot.executed_trades

        assert recorder.snapshot_counts() == baseline_counts
    finally:
        for handle in handles:
            event_bus.unsubscribe(handle)
        if event_bus.is_running():
            await event_bus.stop()
