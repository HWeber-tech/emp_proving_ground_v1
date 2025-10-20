"""Bootstrap runtime bridging the sensory stack with the trading manager."""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Coroutine, Deque, Mapping, Sequence

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
from src.trading.execution.paper_broker_adapter import PaperBrokerExecutionAdapter
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.execution.trade_throttle import TradeThrottleConfig
from src.trading.liquidity.depth_aware_prober import DepthAwareLiquidityProber
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor, RedisLike
from src.trading.gating import DriftSentryGate
from src.trading.trading_manager import TradingManager
from src.governance.policy_ledger import LedgerReleaseManager
from src.understanding.decision_diary import DecisionDiaryStore

logger = logging.getLogger(__name__)


def _normalise_anchor(anchor: datetime | None) -> datetime:
    if anchor is None:
        return datetime.now(timezone.utc)
    if anchor.tzinfo is None:
        return anchor.replace(tzinfo=timezone.utc)
    return anchor.astimezone(timezone.utc)


_REPLAY_SERIES_LENGTH = 144
_REPLAY_WARMUP_STEPS = 5
_REPLAY_MAIN_WEIGHT = 0.8
_REPLAY_RARE_WEIGHT = 0.2
_REPLAY_TEMPERATURE_START = 1.2
_REPLAY_TEMPERATURE_MIN = 0.8
_REPLAY_TEMPERATURE_DECAY = 0.995


def _temperature_mix(
    temperature: float,
    *,
    main_weight: float = _REPLAY_MAIN_WEIGHT,
    rare_weight: float = _REPLAY_RARE_WEIGHT,
) -> tuple[float, float]:
    main_scaled = main_weight ** (1.0 / temperature)
    rare_scaled = rare_weight ** (1.0 / temperature)
    total = main_scaled + rare_scaled
    if total <= 0:
        return 1.0, 0.0
    return main_scaled / total, rare_scaled / total


class _ReplaySeriesBuilder:
    """Synthesize bootstrap replay buffers with rare regime injections."""

    def __init__(
        self,
        symbol: str,
        *,
        index: int,
        anchor: datetime,
    ) -> None:
        self._symbol = symbol
        self._symbol_index = index
        self._anchor = anchor
        self._base_price = 1.10 + 0.002 * index
        self._amplitude = 0.0008 + 0.0002 * index
        self._drift = 0.0003 + 0.00005 * index
        self._depth_seed = 5200 + index * 250
        self._imbalance_bias = 0.15 + 0.05 * index
        self._phase = 0.6 * (index + 1)
        self._price = self._base_price
        self._prev_price = self._base_price
        self._step = 0
        self._temperature = _REPLAY_TEMPERATURE_START
        self._rare_cycle = ("rare_nfp", "rare_halt")
        self._rare_index = 0
        self._nfp_direction = 1.0

    def build(self, length: int) -> list[MarketData]:
        start = self._anchor - timedelta(minutes=length)
        timestamp = start
        bars: list[MarketData] = []
        for _ in range(length):
            regime = self._select_regime()
            bar = self._next_bar(timestamp, regime)
            bars.append(bar)
            timestamp += timedelta(minutes=1)
            self._step += 1
        return bars

    def _select_regime(self) -> str:
        if self._step < _REPLAY_WARMUP_STEPS:
            return "main"

        main_prob, rare_prob = _temperature_mix(self._temperature)
        threshold = random.random()
        regime = "main" if threshold <= main_prob else self._next_rare_regime()
        self._temperature = max(
            _REPLAY_TEMPERATURE_MIN,
            self._temperature * _REPLAY_TEMPERATURE_DECAY,
        )
        return regime

    def _next_rare_regime(self) -> str:
        regime = self._rare_cycle[self._rare_index % len(self._rare_cycle)]
        self._rare_index += 1
        return regime

    def _base_metrics(self, step: int) -> dict[str, float]:
        volume = 1800.0 + step * 120.0 + self._symbol_index * 200.0
        volatility = 0.00035 + 0.00005 * abs(math.sin((self._phase + step) / 2.3))
        spread = 0.00005 + 0.00001 * ((self._symbol_index + step) % 4)
        depth = self._depth_seed + step * 180.0
        imbalance = math.tanh((step - 18) / 9.0) * (0.4 + self._imbalance_bias)
        macro_bias = 0.25 + 0.05 * self._symbol_index
        return {
            "volume": volume,
            "volatility": volatility,
            "spread": spread,
            "depth": depth,
            "imbalance": imbalance,
            "macro_bias": macro_bias,
            "data_quality": 0.88,
        }

    def _advance_main(self, step: int) -> float:
        oscillation = self._amplitude * math.sin((self._phase + step) / 3.5)
        self._price = max(0.35, self._price + self._drift + oscillation)
        return self._price

    def _advance_nfp(self) -> float:
        jump = 0.0018 * self._nfp_direction + self._drift * 6.0
        self._price = max(0.35, self._price + jump)
        self._nfp_direction *= -1.0
        return self._price

    def _advance_halt(self) -> float:
        self._price = max(0.35, self._price + self._drift * 0.15)
        return self._price

    def _next_bar(self, timestamp: datetime, regime: str) -> MarketData:
        metrics = self._base_metrics(self._step)
        if regime == "main":
            price = self._advance_main(self._step)
        elif regime == "rare_nfp":
            price = self._advance_nfp()
            metrics["volume"] *= 2.8
            metrics["volatility"] *= 3.5
            metrics["spread"] *= 2.4
            metrics["imbalance"] *= 1.35
            metrics["macro_bias"] += 0.06 * self._nfp_direction
            metrics["data_quality"] = 0.83
        else:  # rare_halt
            price = self._advance_halt()
            metrics["volume"] = max(40.0, metrics["volume"] * 0.08)
            metrics["volatility"] = max(0.00002, metrics["volatility"] * 0.25)
            metrics["spread"] *= 3.6
            metrics["depth"] *= 0.45
            metrics["imbalance"] *= 0.25
            metrics["macro_bias"] -= 0.08
            metrics["data_quality"] = 0.62

        open_price = self._prev_price
        tick_range = max(0.00005, metrics["volatility"] * 6.0)
        high_price = max(open_price, price) + tick_range
        low_price = min(open_price, price) - tick_range

        bar = MarketData(
            symbol=self._symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=price,
            volume=metrics["volume"],
            volatility=metrics["volatility"],
            spread=metrics["spread"],
            depth=metrics["depth"],
            order_imbalance=metrics["imbalance"],
            macro_bias=metrics["macro_bias"],
            data_quality=metrics["data_quality"],
            replay_regime=regime,
            replay_temperature=self._temperature,
        )
        self._prev_price = price
        return bar


def _generate_bootstrap_series(
    symbols: Sequence[str],
    *,
    anchor: datetime | None = None,
) -> Mapping[str, list[MarketData]]:
    """Build a deterministic synthetic price series for bootstrap deployments."""

    now = _normalise_anchor(anchor)
    series: dict[str, list[MarketData]] = {}

    for idx, symbol in enumerate(symbols or ("EURUSD",)):
        builder = _ReplaySeriesBuilder(symbol, index=idx, anchor=now)
        series[symbol] = builder.build(_REPLAY_SERIES_LENGTH)

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
        diary_store: DecisionDiaryStore | None = None,
        trade_throttle: TradeThrottleConfig | Mapping[str, object] | None = None,
        series_anchor: datetime | None = None,
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
        supervisor = task_supervisor
        owns_supervisor = False
        if supervisor is None:
            supervisor = TaskSupervisor(namespace="bootstrap-runtime")
            owns_supervisor = True
        self._task_supervisor = supervisor
        self._owns_supervisor = owns_supervisor
        self._task_metadata: dict[asyncio.Task[Any], dict[str, Any]] = {}

        resolved_connectors: dict[str, MarketDataConnector] = {}
        if connectors:
            resolved_connectors.update({str(k): v for k, v in connectors.items()})
        if "historical_replay" not in resolved_connectors:
            resolved_connectors["historical_replay"] = HistoricalReplayConnector(
                _generate_bootstrap_series(self.symbols, anchor=series_anchor)
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
        self._diary_store = diary_store

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
            trade_throttle=trade_throttle,
        )
        self.portfolio_monitor: PortfolioMonitor = self.trading_manager.portfolio_monitor
        self.execution_engine = ImmediateFillExecutionAdapter(self.portfolio_monitor)
        self.trading_manager.execution_engine = self.execution_engine
        self._release_router: ReleaseAwareExecutionRouter | None = None
        self._paper_broker_adapter: Any | None = None
        self._paper_broker_summary: Mapping[str, Any] | None = None
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
            diary_store=diary_store,
            release_router=self._release_router,
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
        try:
            status["pipeline_observability"] = self.trading_stack.describe_pipeline_observability()
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to capture pipeline observability snapshot", exc_info=True)
        if self._paper_broker_summary:
            status["paper_broker"] = dict(self._paper_broker_summary)
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

        supervisor = self._task_supervisor
        if supervisor is not None:
            status["background_tasks"] = {
                "count": supervisor.active_count,
                "tasks": self.describe_background_tasks(),
            }
        return status

    def describe_paper_broker(self) -> Mapping[str, Any] | None:
        """Expose the configured paper trading adapter summary, if any."""

        summary: dict[str, Any] = {}
        if isinstance(self._paper_broker_summary, Mapping):
            summary.update(
                {str(key): value for key, value in self._paper_broker_summary.items()}
            )
        paper_engine = getattr(self.trading_manager, "_live_engine", None)
        if isinstance(paper_engine, PaperBrokerExecutionAdapter):
            summary.setdefault("metrics", paper_engine.describe_metrics())
            risk_snapshot = paper_engine.describe_risk_context()
            if risk_snapshot:
                summary.setdefault("risk_context", risk_snapshot)
            last_error = paper_engine.describe_last_error()
            if isinstance(last_error, Mapping) and last_error:
                summary.setdefault("last_error", dict(last_error))
            broker_snapshot = paper_engine.describe_broker()
            if isinstance(broker_snapshot, Mapping) and broker_snapshot:
                for key, value in broker_snapshot.items():
                    summary.setdefault(str(key), value)
        if not summary:
            return None
        return summary

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
        if supervisor is None:  # pragma: no cover - defensive guard
            supervisor = TaskSupervisor(namespace="bootstrap-runtime")
            self._task_supervisor = supervisor
            self._owns_supervisor = True

        attach_supervisor = getattr(self.trading_manager, "attach_task_supervisor", None)
        if callable(attach_supervisor):
            try:
                attach_supervisor(supervisor)
            except Exception:  # pragma: no cover - diagnostic guardrail
                logger.debug(
                    "Failed to bind task supervisor to trading manager",
                    exc_info=True,
                )

        drift_config = self._sensory_drift_config
        metadata = {
            "component": "understanding.loop",
            "symbols": tuple(self.symbols),
            "tick_interval": float(self.tick_interval),
            "max_ticks": self.max_ticks,
            "drift_window": drift_config.required_samples(),
            "drift_sensors": tuple(drift_config.sensors),
            "evolution_interval": self._evolution_cycle_interval,
        }

        self._run_task = self.create_background_task(
            self._run_loop(),
            name="bootstrap-runtime-loop",
            metadata=metadata,
            restart_callback=self._run_loop,
            max_restarts=None,
            restart_backoff=self.tick_interval if self.tick_interval > 0 else 0.0,
        )
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
        if task is not None:
            self._task_metadata.pop(task, None)
        self._run_task = None
        self._price_task = None
        if self._owns_supervisor and self._task_supervisor is not None:
            await self._task_supervisor.cancel_all()
        logger.info("BootstrapRuntime stopped after %s ticks", self._tick_counter)

    @property
    def task_supervisor(self) -> TaskSupervisor | None:
        """Expose the task supervisor coordinating bootstrap background work."""

        return self._task_supervisor

    def create_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        restart_callback: Callable[[], Coroutine[Any, Any, Any]] | None = None,
        max_restarts: int | None = 0,
        restart_backoff: float = 0.0,
        hang_timeout: float | None = None,
    ) -> asyncio.Task[Any]:
        """Create and track a supervised background task bound to the runtime."""

        if not asyncio.iscoroutine(coro):
            raise TypeError("BootstrapRuntime.create_background_task expects a coroutine")

        supervisor = self._task_supervisor
        if supervisor is None:
            supervisor = TaskSupervisor(namespace="bootstrap-runtime")
            self._task_supervisor = supervisor
            self._owns_supervisor = True

        metadata_payload = dict(metadata) if metadata is not None else None
        task = supervisor.create(
            coro,
            name=name,
            metadata=metadata_payload,
            restart_callback=restart_callback,
            max_restarts=max_restarts,
            restart_backoff=restart_backoff,
            hang_timeout=hang_timeout,
        )
        self._register_task_metadata(task, metadata_payload)
        return task

    def register_background_task(
        self,
        task: asyncio.Task[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register an externally created task with the runtime supervisor."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("BootstrapRuntime.register_background_task expects an asyncio.Task")

        supervisor = self._task_supervisor
        if supervisor is None:
            supervisor = TaskSupervisor(namespace="bootstrap-runtime")
            self._task_supervisor = supervisor
            self._owns_supervisor = True

        metadata_payload = dict(metadata) if metadata is not None else None
        supervisor.track(task, metadata=metadata_payload)
        self._register_task_metadata(task, metadata_payload)

    def get_background_task_metadata(
        self, task: asyncio.Task[Any]
    ) -> Mapping[str, Any] | None:
        """Expose metadata for registered background tasks."""

        payload = self._task_metadata.get(task)
        if payload is None:
            return None
        return dict(payload)

    def describe_background_tasks(self) -> tuple[dict[str, Any], ...]:
        """Return a serialisable snapshot of active supervised tasks."""

        supervisor = self._task_supervisor
        if supervisor is None:
            return ()
        return tuple(supervisor.describe())

    def _register_task_metadata(
        self,
        task: asyncio.Task[Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        stored = dict(metadata) if metadata is not None else {}
        self._task_metadata[task] = stored
        task.add_done_callback(self._cleanup_task_metadata)

    def _cleanup_task_metadata(self, task: asyncio.Task[Any]) -> None:
        self._task_metadata.pop(task, None)

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
