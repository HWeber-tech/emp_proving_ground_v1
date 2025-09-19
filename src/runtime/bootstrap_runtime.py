"""Bootstrap runtime bridging the sensory stack with the trading manager."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

from src.core.base import MarketData
from src.core.event_bus import EventBus
from src.data_foundation.fabric.historical_connector import HistoricalReplayConnector
from src.data_foundation.fabric.market_data_fabric import MarketDataConnector, MarketDataFabric
from src.operations.bootstrap_control_center import BootstrapControlCenter
from src.orchestration.bootstrap_stack import BootstrapSensoryPipeline, BootstrapTradingStack
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.trading_manager import TradingManager

logger = logging.getLogger(__name__)


def _generate_bootstrap_series(symbols: Sequence[str]) -> Mapping[str, list[MarketData]]:
    """Build a deterministic synthetic price series for bootstrap deployments."""

    now = datetime.now(timezone.utc)
    series: dict[str, list[MarketData]] = {}

    for idx, symbol in enumerate(symbols or ("EURUSD",)):
        base = 1.10 + 0.002 * idx
        amplitude = 0.0008 + 0.0002 * idx
        drift = 0.0003 + 0.00005 * idx
        depth_seed = 5200 + idx * 250
        imbalance_bias = 0.15 + 0.05 * idx

        bars: list[MarketData] = []
        price = base
        for step in range(24):
            ts = now - timedelta(minutes=24 - step)
            oscillation = amplitude * math.sin(step / 3.5)
            price = price + drift + oscillation
            open_price = price - 0.00025
            high_price = max(open_price, price) + 0.00018
            low_price = min(open_price, price) - 0.00018

            bar = MarketData(
                symbol=symbol,
                timestamp=ts,
                open=open_price,
                high=high_price,
                low=low_price,
                close=price,
                volume=1800 + step * 120 + idx * 200,
                volatility=0.00035 + 0.00005 * abs(math.sin(step / 2.3)),
                spread=0.00005 + 0.00001 * ((idx + step) % 4),
                depth=depth_seed + step * 180,
                order_imbalance=math.tanh((step - 12) / 9.0) * (0.4 + imbalance_bias),
                macro_bias=0.25 + 0.05 * idx,
                data_quality=0.85,
            )
            bars.append(bar)

        series[symbol] = bars

    return series


class _AlwaysActiveRegistry:
    """Minimal strategy registry returning an active strategy for bootstrap runs."""

    def get_strategy(self, strategy_id: str) -> Mapping[str, Any] | None:
        return {"strategy_id": strategy_id, "status": "active"}


class BootstrapRuntime:
    """Drive the bootstrap sensory, fusion, and trading loop on a schedule."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        symbols: Sequence[str] | None = None,
        connectors: Mapping[str, MarketDataConnector] | None = None,
        tick_interval: float = 2.5,
        max_ticks: int | None = None,
        strategy_id: str = "bootstrap-strategy",
        buy_threshold: float = 0.25,
        sell_threshold: float = 0.25,
        requested_quantity: Decimal | float = Decimal("1"),
        stop_loss_pct: float = 0.01,
        risk_per_trade: float = 0.02,
        max_open_positions: int = 5,
        max_daily_drawdown: float = 0.1,
        initial_equity: float = 100_000.0,
        min_intent_confidence: float = 0.05,
        min_liquidity_confidence: float = 0.25,
        liquidity_prober: DepthAwareLiquidityProber | None = None,
        evolution_orchestrator: Any | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.symbols = [s.strip() for s in (symbols or ["EURUSD"]) if s and s.strip()]
        if not self.symbols:
            self.symbols = ["EURUSD"]
        self.tick_interval = max(0.0, float(tick_interval))
        self.max_ticks = None if max_ticks is None else max(0, int(max_ticks))
        self._tick_counter = 0
        self._stop_event = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None
        self._price_task: asyncio.Task[None] | None = None
        self.running = False
        self._last_error: Exception | None = None

        resolved_connectors: dict[str, MarketDataConnector]
        if connectors:
            resolved_connectors = {str(k): v for k, v in connectors.items()}
        else:
            resolved_connectors = {
                "historical_replay": HistoricalReplayConnector(_generate_bootstrap_series(self.symbols))
            }

        self.fabric = MarketDataFabric(resolved_connectors)
        self.pipeline = BootstrapSensoryPipeline(self.fabric)

        self.liquidity_prober = liquidity_prober or DepthAwareLiquidityProber()
        self.evolution_orchestrator = evolution_orchestrator

        self.trading_manager = TradingManager(
            event_bus=event_bus,
            strategy_registry=_AlwaysActiveRegistry(),
            execution_engine=None,
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
            max_open_positions=max_open_positions,
            max_daily_drawdown=max_daily_drawdown,
            redis_client=InMemoryRedis(),
            liquidity_prober=self.liquidity_prober,
            min_intent_confidence=min_intent_confidence,
            min_liquidity_confidence=min_liquidity_confidence,
        )
        self.portfolio_monitor: PortfolioMonitor = self.trading_manager.portfolio_monitor
        self.execution_engine = ImmediateFillExecutionAdapter(self.portfolio_monitor)
        self.trading_manager.execution_engine = self.execution_engine

        self.control_center = BootstrapControlCenter(
            pipeline=self.pipeline,
            trading_manager=self.trading_manager,
            execution_adapter=self.execution_engine,
            liquidity_prober=self.liquidity_prober,
            evolution_orchestrator=self.evolution_orchestrator,
        )

        self.trading_stack = BootstrapTradingStack(
            pipeline=self.pipeline,
            trading_manager=self.trading_manager,
            strategy_id=strategy_id,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            requested_quantity=requested_quantity,
            stop_loss_pct=stop_loss_pct,
            liquidity_prober=self.liquidity_prober,
            control_center=self.control_center,
        )

    @property
    def decisions(self) -> list[dict[str, Any]]:
        return self.trading_stack.decisions

    def status(self) -> Mapping[str, Any]:
        telemetry = self.control_center.overview()
        status = {
            "running": self.running,
            "symbols": list(self.symbols),
            "tick_interval": self.tick_interval,
            "ticks_processed": self._tick_counter,
            "decisions": len(self.trading_stack.decisions),
            "fills": len(self.execution_engine.fills),
            "last_error": self._last_error.__class__.__name__ if self._last_error else None,
            "telemetry": telemetry,
        }
        vision_summary = telemetry.get("vision_alignment") if isinstance(telemetry, Mapping) else None
        if isinstance(vision_summary, Mapping):
            status["vision_alignment"] = vision_summary
        return status

    async def start(self) -> None:
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._tick_counter = 0
        self.running = True
        self._run_task = asyncio.create_task(self._run_loop(), name="bootstrap-runtime-loop")
        self._price_task = self._run_task
        logger.info(
            "BootstrapRuntime started for %s (tick interval %.2fs)",
            ",".join(self.symbols),
            self.tick_interval,
        )

    async def stop(self) -> None:
        if not self.running:
            return
        self._stop_event.set()
        task = self._run_task
        if task and not task.done():
            await task
        self.running = False
        self._run_task = None
        self._price_task = None
        logger.info("BootstrapRuntime stopped after %s ticks", self._tick_counter)

    async def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                for symbol in self.symbols:
                    if self._stop_event.is_set():
                        break
                    try:
                        await self.trading_stack.evaluate_tick(symbol)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        self._last_error = exc
                        logger.warning("BootstrapRuntime tick failure for %s: %s", symbol, exc)
                self._tick_counter += 1
                if self.max_ticks is not None and self._tick_counter >= self.max_ticks:
                    break
                if self.tick_interval == 0:
                    await asyncio.sleep(0)
                else:
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=self.tick_interval)
                    except asyncio.TimeoutError:
                        continue
        finally:
            self.running = False
            self._stop_event.set()
            self._run_task = None
            self._price_task = None


__all__ = ["BootstrapRuntime"]
