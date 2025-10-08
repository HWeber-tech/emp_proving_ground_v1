"""Bootstrap runtime bridging the sensory stack with the trading manager."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Deque, Mapping, Sequence

import pandas as pd

from src.config.risk.risk_config import RiskConfig
from src.core.base import MarketData
from src.core.event_bus import EventBus
from src.data_foundation.cache import ManagedRedisCache, RedisCachePolicy, wrap_managed_cache
from src.data_foundation.fabric.historical_connector import HistoricalReplayConnector
from src.data_foundation.fabric.market_data_fabric import MarketDataConnector, MarketDataFabric
from src.operations.bootstrap_control_center import BootstrapControlCenter
from src.operations.event_bus_failover import EventPublishError
from src.operations.roi import RoiCostModel
from src.operations.sensory_drift import evaluate_sensory_drift, publish_sensory_drift
from src.operations.sensory_metrics import build_sensory_metrics, publish_sensory_metrics
from src.operations.sensory_summary import build_sensory_summary, publish_sensory_summary
from src.orchestration.bootstrap_stack import BootstrapSensoryPipeline, BootstrapTradingStack
from src.sensory.lineage_publisher import SensoryLineagePublisher
from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig
from src.runtime.task_supervisor import TaskSupervisor
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor, RedisLike
from src.trading.gating import DriftSentryGate
from src.trading.trading_manager import TradingManager
from src.governance.policy_ledger import LedgerReleaseManager

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
        risk_per_trade: float | None = None,
        max_open_positions: int = 5,
        max_daily_drawdown: float | None = None,
        initial_equity: float = 100_000.0,
        min_intent_confidence: float = 0.05,
        min_liquidity_confidence: float = 0.25,
        liquidity_prober: DepthAwareLiquidityProber | None = None,
        evolution_orchestrator: Any | None = None,
        evolution_cycle_interval: int = 5,
        redis_client: RedisLike | None = None,
        roi_cost_model: RoiCostModel | None = None,
        risk_config: RiskConfig | None = None,
        task_supervisor: TaskSupervisor | None = None,
        release_manager: LedgerReleaseManager | None = None,
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
        self._task_supervisor: TaskSupervisor | None = task_supervisor
        self._owns_supervisor = task_supervisor is None

        resolved_connectors: dict[str, MarketDataConnector] = {}
        if connectors:
            resolved_connectors.update({str(k): v for k, v in connectors.items()})
        if "historical_replay" not in resolved_connectors:
            resolved_connectors["historical_replay"] = HistoricalReplayConnector(
                _generate_bootstrap_series(self.symbols)
            )

        self.fabric = MarketDataFabric(resolved_connectors)
        self.pipeline = BootstrapSensoryPipeline(self.fabric)

        self.liquidity_prober = liquidity_prober or DepthAwareLiquidityProber()
        self.evolution_orchestrator = evolution_orchestrator
        self._evolution_cycle_interval = max(1, int(evolution_cycle_interval or 1))

        if isinstance(redis_client, ManagedRedisCache):
            cache_client = redis_client
        else:
            default_policy = RedisCachePolicy.bootstrap_defaults()
            cache_client = wrap_managed_cache(
                redis_client, policy=default_policy, bootstrap=redis_client is None
            )

        self.redis_client = cache_client
        self._release_manager = release_manager

        base_risk_per_trade = risk_per_trade if risk_per_trade is not None else 0.02
        base_drawdown = max_daily_drawdown if max_daily_drawdown is not None else 0.1
        resolved_risk_config = risk_config or RiskConfig(
            max_risk_per_trade_pct=Decimal(str(base_risk_per_trade)),
            max_total_exposure_pct=Decimal("0.5"),
            max_leverage=Decimal("10.0"),
            max_drawdown_pct=Decimal(str(base_drawdown)),
            min_position_size=1,
        )

        self._drift_gate = DriftSentryGate()
        self.trading_manager = TradingManager(
            event_bus=event_bus,
            strategy_registry=_AlwaysActiveRegistry(),
            execution_engine=None,
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
            max_open_positions=max_open_positions,
            max_daily_drawdown=max_daily_drawdown,
            redis_client=cache_client,
            liquidity_prober=self.liquidity_prober,
            task_supervisor=self._task_supervisor,
            min_intent_confidence=min_intent_confidence,
            min_liquidity_confidence=min_liquidity_confidence,
            roi_cost_model=roi_cost_model,
            risk_config=resolved_risk_config,
            drift_gate=self._drift_gate,
            release_manager=release_manager,
        )
        self.portfolio_monitor: PortfolioMonitor = self.trading_manager.portfolio_monitor
        self.execution_engine = ImmediateFillExecutionAdapter(self.portfolio_monitor)
        self.trading_manager.execution_engine = self.execution_engine
        self._release_router: ReleaseAwareExecutionRouter | None = None
        if release_manager is not None:
            try:
                default_stage = release_manager.resolve_stage(strategy_id)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to resolve initial release stage for bootstrap runtime",
                    exc_info=True,
                )
                default_stage = None
            try:
                self._release_router = self.trading_manager.install_release_execution_router(
                    paper_engine=self.execution_engine,
                    pilot_engine=self.execution_engine,
                    live_engine=self.execution_engine,
                    default_stage=default_stage,
                )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to install release-aware execution router",
                    exc_info=True,
                )
                self._release_router = None

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

        drift_config = SensoryDriftConfig(
            baseline_window=24,
            evaluation_window=12,
            min_observations=6,
            sensors=("WHY", "WHAT", "WHEN", "HOW", "ANOMALY"),
        )
        self._sensory_drift_config = drift_config
        self._sensory_lineage_publisher = SensoryLineagePublisher(event_bus=event_bus)
        self._sensory_lineage_history_limit = 15
        self._sensory_cortex = RealSensoryOrgan(
            event_bus=event_bus,
            drift_config=drift_config,
            lineage_publisher=self._sensory_lineage_publisher,
        )
        self._sensory_history_window = 256
        self._sensory_history: dict[str, Deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._sensory_history_window)
        )
        self._latest_sensory_metrics: Mapping[str, Any] | None = None

    @property
    def decisions(self) -> list[dict[str, Any]]:
        return self.trading_stack.decisions

    def _market_data_to_record(self, data: MarketData) -> dict[str, Any]:
        record = {
            "timestamp": getattr(data, "timestamp", datetime.now(timezone.utc)),
            "symbol": getattr(data, "symbol", "UNKNOWN"),
            "open": float(getattr(data, "open", 0.0) or 0.0),
            "high": float(getattr(data, "high", 0.0) or 0.0),
            "low": float(getattr(data, "low", 0.0) or 0.0),
            "close": float(getattr(data, "close", 0.0) or 0.0),
            "volume": float(getattr(data, "volume", 0.0) or 0.0),
            "volatility": float(getattr(data, "volatility", 0.0) or 0.0),
            "spread": float(getattr(data, "spread", 0.0) or 0.0),
            "depth": float(getattr(data, "depth", 0.0) or 0.0),
            "order_imbalance": float(getattr(data, "order_imbalance", 0.0) or 0.0),
            "data_quality": float(getattr(data, "data_quality", 0.85) or 0.85),
            "macro_bias": float(getattr(data, "macro_bias", 0.0) or 0.0),
        }

        for yield_attr in ("yield_curve", "yield_2y", "yield_5y", "yield_10y", "yield_30y"):
            value = getattr(data, yield_attr, None)
            if value is not None:
                record[yield_attr] = value

        return record

    def _update_sensory_observation(self, snapshot: "SensorySnapshot") -> None:
        try:
            record = self._market_data_to_record(snapshot.market_data)
            history = self._sensory_history[snapshot.symbol]
            history.append(record)
            frame = pd.DataFrame(list(history))
            metadata = {
                "runtime": "bootstrap",
                "tick": self._tick_counter,
                "strategy_id": getattr(self.trading_stack, "strategy_id", None),
            }
            self._sensory_cortex.observe(
                frame,
                symbol=snapshot.symbol,
                as_of=record["timestamp"],
                metadata=metadata,
            )
            self._publish_sensory_outputs(snapshot)
        except Exception:  # pragma: no cover - defensive guard for telemetry path
            logger.debug("Failed to update sensory cortex observation", exc_info=True)

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
            "evolution_cycle_interval": self._evolution_cycle_interval,
        }
        vision_summary = (
            telemetry.get("vision_alignment") if isinstance(telemetry, Mapping) else None
        )
        if isinstance(vision_summary, Mapping):
            status["vision_alignment"] = vision_summary
        pipeline_audit: list[Mapping[str, Any]] = []
        if hasattr(self.pipeline, "audit_trail"):
            try:
                audit_entries = self.pipeline.audit_trail(limit=5)
            except Exception:  # pragma: no cover - diagnostics must not break status
                audit_entries = []
            if audit_entries:
                pipeline_audit = [
                    dict(entry) for entry in audit_entries if isinstance(entry, Mapping)
                ]
                status["legacy_sensor_audit"] = pipeline_audit

        sensory_status = self._sensory_cortex.status()
        if isinstance(sensory_status, Mapping):
            status["sensory_cortex"] = sensory_status
            try:
                status["samples"] = int(sensory_status.get("samples") or 0)
            except Exception:
                status["samples"] = 0

            latest_payload = sensory_status.get("latest")
            if isinstance(latest_payload, Mapping):
                status["latest"] = dict(latest_payload)

            drift_summary = sensory_status.get("drift_summary")
            if isinstance(drift_summary, Mapping):
                status["drift_summary"] = drift_summary

            audit_payload = sensory_status.get("sensor_audit")
            if isinstance(audit_payload, list) and audit_payload:
                status["sensor_audit"] = [
                    dict(entry) if isinstance(entry, Mapping) else entry
                    for entry in audit_payload
                ]
            elif pipeline_audit:
                status["sensor_audit"] = pipeline_audit

            metrics_payload = self._latest_sensory_metrics or self._sensory_cortex.metrics()
            if isinstance(metrics_payload, Mapping):
                status["sensory_metrics"] = metrics_payload

            if pipeline_audit:
                status.setdefault("legacy_sensor_audit", pipeline_audit)

            if self._sensory_lineage_publisher is not None:
                try:
                    lineage_history = self._sensory_lineage_publisher.history(
                        limit=self._sensory_lineage_history_limit
                    )
                except Exception:  # pragma: no cover - defensive telemetry guard
                    logger.debug(
                        "Failed to collect sensory lineage history for bootstrap status",
                        exc_info=True,
                    )
                else:
                    if lineage_history:
                        status["sensory_lineage"] = lineage_history
                        status["sensory_lineage_latest"] = lineage_history[0]
        elif pipeline_audit:
            status["sensor_audit"] = pipeline_audit

        describe_interface = getattr(self.trading_manager, "describe_risk_interface", None)
        if callable(describe_interface):
            try:
                interface_payload = describe_interface()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to resolve trading risk interface for bootstrap status",
                    exc_info=True,
                )
            else:
                if interface_payload is not None:
                    if isinstance(interface_payload, Mapping):
                        status["risk_interface"] = dict(interface_payload)
                    else:
                        status["risk_interface"] = interface_payload

        describe_release = getattr(self.trading_manager, "describe_release_posture", None)
        if callable(describe_release):
            try:
                strategy_id = getattr(self.trading_stack, "strategy_id", None)
                release_payload = describe_release(strategy_id)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to resolve policy release posture for bootstrap status",
                    exc_info=True,
                )
            else:
                if release_payload:
                    status["release_posture"] = dict(release_payload)
        describe_execution = getattr(self.trading_manager, "describe_release_execution", None)
        if callable(describe_execution):
            try:
                release_execution = describe_execution()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to resolve release execution routing for bootstrap status",
                    exc_info=True,
                )
            else:
                if isinstance(release_execution, Mapping):
                    status["release_execution"] = dict(release_execution)
                elif release_execution is not None:
                    status["release_execution"] = release_execution
        return status

    def _publish_sensory_outputs(self, snapshot: Mapping[str, Any] | None) -> None:
        status_payload: Mapping[str, Any] | None
        try:
            status_payload = self._sensory_cortex.status()
        except Exception:
            logger.debug("Failed to capture sensory status for bootstrap telemetry", exc_info=True)
            status_payload = None

        metrics_payload = None
        try:
            metrics_payload = self._sensory_cortex.metrics()
        except Exception:
            logger.debug(
                "Failed to capture sensory metrics payload prior to publishing",
                exc_info=True,
            )

        if isinstance(metrics_payload, Mapping):
            self._latest_sensory_metrics = metrics_payload
        else:
            self._latest_sensory_metrics = None

        summary = None
        try:
            summary = build_sensory_summary(status_payload)
        except Exception:
            logger.debug(
                "Failed to build sensory summary for bootstrap runtime",
                exc_info=True,
            )
        else:
            try:
                publish_sensory_summary(summary, event_bus=self.event_bus)
            except EventPublishError:
                logger.debug(
                    "Failed to publish sensory summary during bootstrap runtime",
                    exc_info=True,
                )
            except Exception:
                logger.debug(
                    "Unexpected error publishing sensory summary during bootstrap runtime",
                    exc_info=True,
                )

            try:
                metrics_dataclass = build_sensory_metrics(summary)
            except Exception:
                logger.debug(
                    "Failed to build sensory metrics from summary during bootstrap runtime",
                    exc_info=True,
                )
            else:
                self._latest_sensory_metrics = metrics_dataclass.as_dict()
                try:
                    publish_sensory_metrics(metrics_dataclass, event_bus=self.event_bus)
                except EventPublishError:
                    logger.debug(
                        "Failed to publish sensory metrics during bootstrap runtime",
                        exc_info=True,
                    )
                except Exception:
                    logger.debug(
                        "Unexpected error publishing sensory metrics during bootstrap runtime",
                        exc_info=True,
                    )

        audit_entries: list[Mapping[str, Any]]
        try:
            audit_entries = self._sensory_cortex.audit_trail(
                limit=self._sensory_drift_config.required_samples()
            )
        except Exception:
            logger.debug(
                "Failed to collect sensory audit trail for drift telemetry",
                exc_info=True,
            )
            return

        if not audit_entries:
            return

        drift_metadata: dict[str, Any] = {
            "runtime": "bootstrap",
            "tick": self._tick_counter,
        }

        if isinstance(snapshot, Mapping):
            symbol = snapshot.get("symbol")
            if symbol is not None:
                drift_metadata["symbol"] = symbol
            generated_at = snapshot.get("generated_at")
            if generated_at is not None:
                drift_metadata["generated_at"] = generated_at

        try:
            drift_snapshot = evaluate_sensory_drift(
                audit_entries,
                lookback=self._sensory_drift_config.required_samples(),
                metadata=drift_metadata,
            )
        except Exception:
            logger.debug(
                "Failed to evaluate sensory drift telemetry during bootstrap runtime",
                exc_info=True,
            )
            return

        try:
            publish_sensory_drift(self.event_bus, drift_snapshot)
        except EventPublishError:
            logger.debug(
                "Failed to publish sensory drift telemetry during bootstrap runtime",
                exc_info=True,
            )
        except Exception:
            logger.debug(
                "Unexpected error publishing sensory drift telemetry during bootstrap runtime",
                exc_info=True,
            )

    async def start(self, *, task_supervisor: TaskSupervisor | None = None) -> None:
        if self.running:
            return
        self._stop_event = asyncio.Event()
        self._tick_counter = 0
        self.running = True
        if task_supervisor is not None:
            self._task_supervisor = task_supervisor
            self._owns_supervisor = False

        supervisor = self._task_supervisor
        if supervisor is None:
            supervisor = TaskSupervisor(namespace="bootstrap-runtime")
            self._task_supervisor = supervisor
            self._owns_supervisor = True

        self._run_task = supervisor.create(self._run_loop(), name="bootstrap-runtime-loop")
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
        if self._owns_supervisor and self._task_supervisor is not None:
            await self._task_supervisor.cancel_all()
        logger.info("BootstrapRuntime stopped after %s ticks", self._tick_counter)

    async def _run_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                for symbol in self.symbols:
                    if self._stop_event.is_set():
                        break
                    try:
                        result = await self.trading_stack.evaluate_tick(symbol)
                        snapshot = (
                            result.get("snapshot")
                            if isinstance(result, Mapping)
                            else None
                        )
                        if snapshot is not None:
                            self._update_sensory_observation(snapshot)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        self._last_error = exc
                        logger.warning("BootstrapRuntime tick failure for %s: %s", symbol, exc)
                should_run_evolution = False
                if self.evolution_orchestrator is not None:
                    if self._tick_counter == 0:
                        should_run_evolution = True
                    elif self._tick_counter % self._evolution_cycle_interval == 0:
                        should_run_evolution = True
                if should_run_evolution:
                    try:
                        await self.evolution_orchestrator.run_cycle()
                    except Exception:  # pragma: no cover - defensive guard for evolution path
                        logger.warning(
                            "Bootstrap evolution orchestrator cycle failed",
                            exc_info=True,
                        )
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
