"""Trading Manager v1.0 - Risk-Aware Trade Execution Coordinator."""

import logging
import time
from collections import deque
from collections.abc import Mapping as MappingABC, MutableMapping as MutableMappingABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast
from uuid import uuid4

from pydantic import ValidationError

try:  # pragma: no cover - redis optional in bootstrap deployments
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from src.config.risk.risk_config import RiskConfig as TradingRiskConfig
from src.compliance.workflow import ComplianceWorkflowSnapshot
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.core.coercion import coerce_float, coerce_int
from src.risk import RiskManager, create_risk_manager
from src.risk.telemetry import (
    RiskTelemetrySnapshot,
    evaluate_risk_posture,
    format_risk_markdown,
    publish_risk_snapshot,
)
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.policy_telemetry import (
    RiskPolicyEvaluationSnapshot,
    RiskPolicyViolationAlert,
    RISK_POLICY_VIOLATION_RUNBOOK,
    build_policy_snapshot,
    build_policy_violation_alert,
    format_policy_markdown,
    format_policy_violation_markdown,
    publish_policy_snapshot,
    publish_policy_violation,
)
from src.trading.risk.risk_gateway import RiskGateway
from src.trading.risk.risk_policy import RiskPolicy
from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    RiskApiError,
    build_runtime_risk_metadata,
    merge_risk_references,
    resolve_trading_risk_interface,
    summarise_risk_config,
)
from src.trading.risk.risk_interface_telemetry import (
    RiskInterfaceErrorAlert,
    RiskInterfaceSnapshot,
    build_risk_interface_error,
    build_risk_interface_snapshot,
    format_risk_interface_markdown,
    publish_risk_interface_error,
    publish_risk_interface_snapshot,
)

from src.operations.roi import (
    RoiCostModel,
    RoiTelemetrySnapshot,
    evaluate_roi_posture,
    format_roi_markdown as format_roi_summary,
    publish_roi_snapshot,
)
from src.operations.sensory_drift import SensoryDriftSnapshot
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.execution.backlog_tracker import BacklogObservation, EventBacklogTracker
from src.trading.execution.performance_monitor import ThroughputMonitor
from src.trading.execution.performance_report import (
    build_execution_performance_report,
    build_performance_health_report,
)
from src.trading.execution.resource_monitor import ResourceUsageMonitor
from src.trading.execution.trade_throttle import (
    TradeThrottle,
    TradeThrottleConfig,
    TradeThrottleDecision,
)
from src.trading.execution.paper_broker_adapter import PaperBrokerExecutionAdapter
from src.trading.gating import (
    DriftGateEvent,
    DriftSentryDecision,
    DriftSentryGate,
    ReleaseRouteEvent,
    publish_drift_gate_event,
    publish_release_route_event,
)
from src.trading.gating.adaptive_release import AdaptiveReleaseThresholds

if TYPE_CHECKING:
    from src.runtime.task_supervisor import TaskSupervisor

# Legacy TradeIntent events were previously provided by ``src.core.events``.
# The trading manager accepts heterogeneous payloads so we keep the annotation
# flexible by relying on ``Any`` instead of a removed compatibility shim.
TradeIntent = Any
TradeRejected = Any
from src.risk.position_sizing import position_size as _PositionSizer
# Provide precise callable typing for the sizer.
PositionSizer = cast(Callable[[Decimal, Decimal, Decimal], Decimal], _PositionSizer)

logger = logging.getLogger(__name__)


_ENGINE_UNSET = object()


@dataclass(frozen=True, slots=True)
class TradeIntentOutcome:
    """Structured result emitted after handling a trade intent."""

    status: str
    executed: bool
    throttle: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "executed": self.executed,
        }
        if self.throttle is not None:
            payload["throttle"] = dict(self.throttle)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class _PositionState:
    quantity: float = 0.0
    avg_price: float = 0.0


@dataclass(slots=True)
class StrategyExecutionStats:
    """Aggregated execution telemetry for a single strategy."""

    strategy_id: str
    executed: int = 0
    failed: int = 0
    throttled: int = 0
    rejected: int = 0
    total_notional: float = 0.0
    realized_pnl: float = 0.0
    volume: float = 0.0
    wins: int = 0
    losses: int = 0
    completed_trades: int = 0
    positions: dict[str, _PositionState] = field(default_factory=dict)
    latency_samples: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    last_order_id: str | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def record_success(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        notional: float,
        latency_ms: float | None,
        order_id: Any,
    ) -> None:
        self.executed += 1
        self.total_notional += abs(float(notional))
        self.volume += abs(float(quantity))
        if order_id not in (None, ""):
            self.last_order_id = str(order_id)
        self.last_error = None
        if latency_ms is not None:
            self.latency_samples += 1
            self.total_latency_ms += float(latency_ms)

        realised_delta, closed_trade = self._apply_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )
        self.realized_pnl += realised_delta
        if realised_delta > 1e-9:
            self.wins += 1
        elif realised_delta < -1e-9:
            self.losses += 1
        if closed_trade:
            self.completed_trades += 1
        self.last_updated = datetime.now(tz=timezone.utc)

    def record_failure(self, reason: str | None = None) -> None:
        self.failed += 1
        self.last_error = reason or "execution_failure"
        self.last_updated = datetime.now(tz=timezone.utc)

    def record_throttled(self) -> None:
        self.throttled += 1
        self.last_updated = datetime.now(tz=timezone.utc)

    def record_rejection(self, reason: str | None = None) -> None:
        self.rejected += 1
        if reason:
            self.last_error = reason
        self.last_updated = datetime.now(tz=timezone.utc)

    def avg_latency_ms(self) -> float | None:
        if self.latency_samples == 0:
            return None
        return self.total_latency_ms / self.latency_samples

    def as_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "executed": self.executed,
            "failed": self.failed,
            "throttled": self.throttled,
            "rejected": self.rejected,
            "total_notional": self.total_notional,
            "realized_pnl": self.realized_pnl,
            "volume": self.volume,
            "wins": self.wins,
            "losses": self.losses,
            "completed_trades": self.completed_trades,
            "win_rate": self._win_rate(),
            "roi": self._roi(),
            "last_updated": self.last_updated.isoformat(),
        }
        latency = self.avg_latency_ms()
        if latency is not None:
            payload["avg_latency_ms"] = latency
        if self.last_order_id:
            payload["last_order_id"] = self.last_order_id
        if self.last_error:
            payload["last_error"] = self.last_error

        open_positions = {
            symbol: {
                "quantity": state.quantity,
                "avg_price": state.avg_price,
            }
            for symbol, state in self.positions.items()
            if abs(state.quantity) > 1e-9
        }
        if open_positions:
            payload["open_positions"] = open_positions
        return payload

    def _win_rate(self) -> float | None:
        total = self.wins + self.losses
        if total == 0:
            return None
        return self.wins / total

    def _roi(self) -> float | None:
        if self.total_notional <= 0.0:
            return None
        return self.realized_pnl / self.total_notional

    def _apply_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> tuple[float, bool]:
        qty = float(quantity)
        if qty <= 0:
            return 0.0, False

        state = self.positions.get(symbol)
        if state is None:
            state = _PositionState()
            self.positions[symbol] = state

        position_qty = state.quantity
        avg_price = state.avg_price
        trade_direction = 1.0 if str(side).upper().startswith("B") else -1.0
        trade_qty = trade_direction * qty

        realised = 0.0
        closed_trade = False

        if position_qty == 0.0 or position_qty * trade_qty >= 0.0:
            new_qty = position_qty + trade_qty
            if new_qty != 0.0:
                if position_qty == 0.0:
                    new_avg = float(price)
                else:
                    new_avg = (
                        position_qty * avg_price + trade_qty * float(price)
                    ) / new_qty
            else:
                new_avg = float(price)
        else:
            closing_qty = min(abs(trade_qty), abs(position_qty))
            if closing_qty > 0.0:
                closed_trade = True
                if position_qty > 0.0:
                    realised += (float(price) - avg_price) * closing_qty
                else:
                    realised += (avg_price - float(price)) * closing_qty
            trade_remainder = position_qty + trade_qty
            new_qty = trade_remainder
            if abs(new_qty) < 1e-9:
                new_qty = 0.0
            if position_qty * trade_remainder > 0.0:
                new_avg = avg_price
            elif new_qty == 0.0:
                new_avg = float(price)
            else:
                new_avg = float(price)

        state.quantity = new_qty
        state.avg_price = new_avg
        if abs(state.quantity) < 1e-9:
            # Remove flat positions to keep summaries tidy.
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = state
        return realised, closed_trade

def _coerce_risk_config(
    config: TradingRiskConfig | Mapping[str, object] | None,
) -> TradingRiskConfig:
    if config is None:
        raise ValueError("TradingManager requires a RiskConfig instance")
    if isinstance(config, TradingRiskConfig):
        return config
    if isinstance(config, MappingABC):
        try:
            return TradingRiskConfig.parse_obj(dict(config))
        except ValidationError as exc:
            raise ValueError("Invalid risk_config payload for TradingManager") from exc
    raise TypeError("risk_config must be a TradingRiskConfig or mapping payload")


class TradingManager:
    """
    Coordinates trade execution with integrated risk management.

    This component acts as the central hub for trade processing, ensuring
    all trades pass through the RiskGateway before reaching the execution engine.
    """

    def __init__(
        self,
        event_bus: Any,
        strategy_registry: Any,
        execution_engine: Any,
        initial_equity: float = 10000.0,
        risk_per_trade: float | None = None,
        max_open_positions: int = 5,
        max_daily_drawdown: float | None = None,
        *,
        redis_client: Any | None = None,
        liquidity_prober: Any | None = None,
        task_supervisor: "TaskSupervisor | None" = None,
        min_intent_confidence: float = 0.2,
        min_liquidity_confidence: float = 0.3,
        roi_cost_model: RoiCostModel | None = None,
        risk_config: TradingRiskConfig | Mapping[str, object] | None = None,
        risk_policy: RiskPolicy | None = None,
        drift_gate: DriftSentryGate | None = None,
        release_manager: LedgerReleaseManager | None = None,
        pilot_execution_engine: Any | None = None,
        live_execution_engine: Any | None = None,
        trade_throttle: TradeThrottleConfig | Mapping[str, object] | None = None,
        throughput_monitor: ThroughputMonitor | None = None,
        throughput_window: int | None = None,
        backlog_tracker: EventBacklogTracker | None = None,
        backlog_threshold_ms: float | None = None,
        backlog_window: int | None = None,
        resource_monitor: ResourceUsageMonitor | None = None,
    ) -> None:
        """
        Initialize the TradingManager with risk management components.

        Args:
            event_bus: Event bus for publishing/subscribing to events
            strategy_registry: Registry for strategy status checking
            execution_engine: Engine for executing validated trades
            initial_equity: Starting account equity
            risk_per_trade: Percentage of equity to risk per trade
            max_open_positions: Maximum allowed open positions
            max_daily_drawdown: Maximum daily drawdown percentage
            task_supervisor: Optional supervisor for execution helpers (e.g. probes)
            trade_throttle: Optional configuration limiting trade frequency
            throughput_monitor: Optional shared throughput monitor instance
            throughput_window: Rolling window size for throughput metrics (if monitor not provided)
            backlog_tracker: Optional shared backlog tracker instance
            backlog_threshold_ms: Override for backlog lag threshold in milliseconds
            backlog_window: Rolling window size for backlog tracking (if tracker not provided)
            resource_monitor: Optional resource monitor instance to reuse across managers
        """
        if throughput_monitor is not None and throughput_window is not None:
            raise ValueError(
                "Provide either throughput_monitor or throughput_window, not both"
            )
        if backlog_tracker is not None and (
            backlog_threshold_ms is not None or backlog_window is not None
        ):
            raise ValueError(
                "Provide backlog_tracker or backlog threshold/window overrides, not both"
            )
        if throughput_window is not None and not isinstance(throughput_window, Integral):
            raise TypeError("throughput_window must be an integer when provided")
        if backlog_window is not None and not isinstance(backlog_window, Integral):
            raise TypeError("backlog_window must be an integer when provided")

        self.event_bus = event_bus
        self.strategy_registry = strategy_registry
        self._installing_release_router = False
        self._release_manager = release_manager
        self._adaptive_thresholds: Optional[AdaptiveReleaseThresholds] = (
            AdaptiveReleaseThresholds(release_manager)
            if release_manager is not None
            else None
        )
        self._release_router: ReleaseAwareExecutionRouter | None = None
        self._execution_engine: Any | None = None
        self._pilot_engine: Any | None = pilot_execution_engine
        self._live_engine: Any | None = live_execution_engine
        self.execution_engine = execution_engine

        # Initialize risk management components
        resolved_client = self._resolve_redis_client(redis_client)
        self.portfolio_monitor = PortfolioMonitor(event_bus, resolved_client)
        self.position_sizer = PositionSizer

        base_config = _coerce_risk_config(risk_config)
        resolved_risk_per_trade = (
            float(risk_per_trade)
            if risk_per_trade is not None
            else float(base_config.max_risk_per_trade_pct)
        )
        resolved_drawdown = (
            float(max_daily_drawdown)
            if max_daily_drawdown is not None
            else float(base_config.max_drawdown_pct)
        )

        effective_config = base_config.copy(
            update={
                "max_risk_per_trade_pct": Decimal(str(resolved_risk_per_trade)),
                "max_drawdown_pct": Decimal(str(resolved_drawdown)),
            }
        )
        self._risk_config = effective_config
        self._portfolio_risk_manager: RiskManager = create_risk_manager(
            config=effective_config,
            initial_balance=initial_equity,
        )
        self._risk_policy = risk_policy or RiskPolicy.from_config(effective_config)
        self._last_policy_snapshot: RiskPolicyEvaluationSnapshot | None = None
        self._last_risk_interface_snapshot: RiskInterfaceSnapshot | None = None
        self._last_risk_interface_error: RiskInterfaceErrorAlert | None = None

        self._task_supervisor = task_supervisor
        risk_per_trade_decimal = Decimal(str(resolved_risk_per_trade))
        configured_prober = self._configure_liquidity_prober(liquidity_prober, task_supervisor)
        self.liquidity_prober = configured_prober
        self.risk_gateway = RiskGateway(
            strategy_registry=strategy_registry,
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            risk_per_trade=risk_per_trade_decimal,
            max_open_positions=max_open_positions,
            max_daily_drawdown=resolved_drawdown,
            liquidity_prober=configured_prober,
            min_intent_confidence=min_intent_confidence,
            min_liquidity_confidence=min_liquidity_confidence,
            risk_policy=self._risk_policy,
            portfolio_risk_manager=self._portfolio_risk_manager,
            risk_config=effective_config,
            event_bus=self.event_bus,
        )

        self._last_risk_snapshot: RiskTelemetrySnapshot | None = None
        self._last_roi_snapshot: RoiTelemetrySnapshot | None = None

        base_model = roi_cost_model or RoiCostModel.bootstrap_defaults(initial_equity)
        self._roi_cost_model = base_model
        self._roi_period_start = datetime.now(tz=timezone.utc)
        self._roi_executed_trades = 0
        self._roi_total_notional = 0.0

        self._drift_gate = drift_gate
        self._last_drift_gate_decision: DriftSentryDecision | None = None

        self._maybe_auto_install_release_router()

        throughput_instance = throughput_monitor
        if throughput_instance is None:
            if throughput_window is not None:
                throughput_instance = ThroughputMonitor(window=int(throughput_window))
            else:
                throughput_instance = ThroughputMonitor()
        self._throughput_monitor = throughput_instance

        backlog_instance = backlog_tracker
        if backlog_instance is None:
            backlog_kwargs: dict[str, float | int] = {}
            if backlog_threshold_ms is not None:
                backlog_kwargs["threshold_ms"] = float(backlog_threshold_ms)
            if backlog_window is not None:
                backlog_kwargs["window"] = int(backlog_window)
            backlog_instance = EventBacklogTracker(**backlog_kwargs)
        self._backlog_tracker = backlog_instance

        self._resource_monitor = resource_monitor or ResourceUsageMonitor()

        self._execution_stats: dict[str, object] = {
            "orders_submitted": 0,
            "orders_executed": 0,
            "orders_failed": 0,
            "latency_samples": 0,
            "total_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "last_error": None,
            "last_execution_at": None,
            "last_successful_order": None,
            "throughput": self._throughput_monitor.snapshot(),
            "resource_usage": self._resource_monitor.snapshot(),
            "backlog": self._backlog_tracker.snapshot(),
            "backlog_breaches": 0,
            "last_backlog_breach": None,
            "throttle_blocks": 0,
            "throttle_retry_at": None,
        }
        self._strategy_stats: dict[str, StrategyExecutionStats] = {}
        self._experiment_events: deque[dict[str, Any]] = deque(maxlen=512)

        self._trade_throttle: TradeThrottle | None = None
        self._trade_throttle_snapshot: Mapping[str, Any] | None = None
        if trade_throttle is not None:
            self.configure_trade_throttle(trade_throttle)

        logger.info(
            f"TradingManager initialized with equity={initial_equity}, "
            f"risk_per_trade={resolved_risk_per_trade * 100}%, "
            f"max_open_positions={max_open_positions}, "
            f"max_daily_drawdown={resolved_drawdown * 100}%"
        )
        logger.info("💹 ROI cost model configured: %s", self._roi_cost_model.as_dict())

    @property
    def execution_engine(self) -> Any | None:
        """Return the currently configured execution engine or router."""

        return self._execution_engine

    @execution_engine.setter
    def execution_engine(self, engine: Any | None) -> None:
        self._execution_engine = engine
        if engine is None:
            self._release_router = None
            return

        self._configure_execution_risk_context(engine)

        if isinstance(engine, ReleaseAwareExecutionRouter):
            self._release_router = engine
            return

        if self._installing_release_router or self._release_manager is None:
            return

        try:
            self._installing_release_router = True
            self.install_release_execution_router(
                paper_engine=engine,
                pilot_engine=self._pilot_engine,
                live_engine=self._live_engine,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Auto-install of release-aware execution router failed", exc_info=True)
            self._execution_engine = engine
        finally:
            self._installing_release_router = False

    def apply_risk_config(
        self,
        config: TradingRiskConfig | Mapping[str, object],
        *,
        propagate: bool = True,
    ) -> TradingRiskConfig:
        """Apply ``config`` and refresh dependent risk components."""

        new_config = config
        if not isinstance(new_config, TradingRiskConfig):
            new_config = TradingRiskConfig.parse_obj(dict(new_config))

        self._risk_config = new_config
        self._risk_policy = RiskPolicy.from_config(new_config)
        self._last_policy_snapshot = None

        if propagate:
            try:
                self.risk_gateway.apply_risk_config(new_config, risk_policy=self._risk_policy)
            except Exception:  # pragma: no cover - defensive guard for integration drift
                logger.debug("RiskGateway.apply_risk_config failed", exc_info=True)

            try:
                self._portfolio_risk_manager.update_limits(new_config.dict())
            except Exception:  # pragma: no cover - defensive guard for integration drift
                logger.debug("RiskManager.update_limits failed", exc_info=True)

        return self._risk_config

    def configure_trade_throttle(
        self,
        config: TradeThrottleConfig | Mapping[str, object] | None,
    ) -> None:
        """Configure or disable the trade frequency throttle."""

        if config is None:
            self._trade_throttle = None
            self._trade_throttle_snapshot = None
            self._execution_stats.pop("trade_throttle", None)
            self._execution_stats["throttle_retry_at"] = None
            return

        throttle_config = (
            config
            if isinstance(config, TradeThrottleConfig)
            else TradeThrottleConfig.parse_obj(dict(config))
        )
        self._trade_throttle = TradeThrottle(throttle_config)
        snapshot = self._trade_throttle.snapshot()
        snapshot_dict = dict(snapshot)
        metadata = snapshot_dict.get("metadata")
        if isinstance(metadata, Mapping):
            snapshot_dict["metadata"] = dict(metadata)
        self._trade_throttle_snapshot = snapshot_dict
        self._execution_stats["trade_throttle"] = snapshot_dict
        self._execution_stats["throttle_blocks"] = coerce_int(
            self._execution_stats.get("throttle_blocks"), default=0
        )
        retry_at = snapshot_dict.get("metadata")
        if isinstance(retry_at, Mapping):
            self._execution_stats["throttle_retry_at"] = retry_at.get("retry_at")
        else:
            self._execution_stats["throttle_retry_at"] = None

    def _evaluate_trade_throttle(
        self,
        *,
        strategy_id: str | None,
        symbol: str | None,
        confidence: float | None,
        notional: float | None,
    ) -> TradeThrottleDecision | None:
        if self._trade_throttle is None:
            return None

        context: dict[str, Any] = {}
        if strategy_id:
            context["strategy_id"] = strategy_id
        if symbol:
            context["symbol"] = symbol
        if confidence is not None:
            context["confidence"] = float(confidence)
        if notional is not None:
            context["notional"] = float(notional)

        decision = self._trade_throttle.evaluate(
            now=datetime.now(tz=timezone.utc),
            metadata=context or None,
        )
        snapshot = decision.as_dict()
        metadata_payload = snapshot.get("metadata")
        if isinstance(metadata_payload, Mapping):
            snapshot["metadata"] = dict(metadata_payload)
        self._trade_throttle_snapshot = snapshot
        self._execution_stats["trade_throttle"] = snapshot
        if decision.retry_at is not None:
            retry_iso = decision.retry_at.astimezone(timezone.utc).isoformat()
            self._execution_stats["throttle_retry_at"] = retry_iso
            snapshot.setdefault("metadata", {}).setdefault("retry_at", retry_iso)
        elif isinstance(snapshot.get("metadata"), Mapping):
            self._execution_stats["throttle_retry_at"] = snapshot["metadata"].get(
                "retry_at"
            )
        else:
            self._execution_stats["throttle_retry_at"] = None

        if not decision.allowed:
            blocks = coerce_int(self._execution_stats.get("throttle_blocks"), default=0)
            self._execution_stats["throttle_blocks"] = blocks + 1
            self._execution_stats["last_throttle_at"] = datetime.now(
                tz=timezone.utc
            ).isoformat()

        return decision

    async def on_trade_intent(self, event: TradeIntent) -> TradeIntentOutcome:
        """
        Handle incoming trade intents with integrated risk management.

        This method replaces the direct execution flow with a risk-validated
        pipeline. All trade intents must pass through the RiskGateway before
        reaching the execution engine.

        Args:
            event: The trade intent event to process
        """
        event_id = getattr(event, "event_id", getattr(event, "id", "unknown"))
        started_wall = datetime.now(tz=timezone.utc)
        gate_decision: DriftSentryDecision | None = None
        drift_event_emitted = False
        drift_event_status: str | None = None
        drift_release_metadata: Mapping[str, Any] | None = None
        drift_confidence_value: float | None = None
        drift_notional_value: float | None = None
        base_symbol: str | None = None
        strategy_id: str | None = None
        base_confidence: float | None = None
        trade_outcome: TradeIntentOutcome | None = None
        try:
            logger.info(f"Received trade intent: {event_id}")

            strategy_id = self._extract_strategy_id(event)
            strategy_stats = self._strategy_stats_for(strategy_id)
            base_symbol = self._extract_symbol(event)
            base_confidence = self._extract_confidence(event)
            side_hint = self._extract_side(event)

            portfolio_state = cast(Any, self.portfolio_monitor).get_state()

            try:
                initial_quantity = self._extract_quantity(event)
                initial_price = self._extract_price(event, portfolio_state)
                drift_notional_estimate = (
                    abs(float(initial_quantity)) * abs(float(initial_price))
                    if initial_price is not None
                    else None
                )
            except Exception:
                drift_notional_estimate = None

            gate_decision = self._evaluate_drift_gate(
                event=event,
                strategy_id=strategy_id,
                symbol=base_symbol,
                confidence=base_confidence,
                portfolio_state=portfolio_state,
                event_id=event_id,
            )
            drift_release_metadata: Mapping[str, Any] | None = None
            drift_event_status: str | None = None
            drift_confidence_value = base_confidence
            drift_notional_value = drift_notional_estimate
            gate_decision_payload = (
                gate_decision.as_dict() if gate_decision is not None else None
            )
            if gate_decision is not None and not gate_decision.allowed:
                logger.warning(
                    "DriftSentry gate forcing paper execution for trade intent %s: %s",
                    event_id,
                    gate_decision.reason,
                )
                notional_for_event: float | None = None
                try:
                    raw_quantity = self._extract_quantity(event)
                    price_estimate = self._extract_price(event, portfolio_state)
                    notional_for_event = abs(float(raw_quantity)) * abs(float(price_estimate))
                except Exception:  # pragma: no cover - diagnostics only
                    notional_for_event = None
                metadata_payload: dict[str, Any] = {
                    "drift_severity": gate_decision.severity.value,
                    "forced_paper": True,
                }
                if gate_decision.reason:
                    metadata_payload["reason"] = gate_decision.reason
                if gate_decision.blocked_dimensions:
                    metadata_payload["blocked_dimensions"] = list(
                        gate_decision.blocked_dimensions
                    )
                if gate_decision.requirements:
                    metadata_payload["requirements"] = dict(gate_decision.requirements)
                if gate_decision.snapshot_metadata:
                    metadata_payload["drift_metadata"] = dict(gate_decision.snapshot_metadata)
                if gate_decision_payload is not None:
                    metadata_payload["drift_gate"] = dict(gate_decision_payload)

                self._record_experiment_event(
                    event_id=event_id,
                    status="forced_paper",
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=base_confidence,
                    notional=notional_for_event,
                    metadata=metadata_payload,
                    decision=None,
                )
                drift_event_status = "forced_paper"
                drift_notional_value = notional_for_event
                drift_confidence_value = base_confidence
                await self._publish_drift_gate_event(
                    decision=gate_decision,
                    event_id=event_id,
                    status=drift_event_status,
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=drift_confidence_value,
                    notional=drift_notional_value,
                    release_metadata=None,
                )

            if gate_decision_payload is not None:
                self._attach_drift_gate_metadata(event, gate_decision_payload)

            validated_intent = await self.risk_gateway.validate_trade_intent(
                intent=event, portfolio_state=portfolio_state
            )

            if validated_intent:
                if gate_decision_payload is not None:
                    self._attach_drift_gate_metadata(validated_intent, gate_decision_payload)
                logger.info(
                    f"Trade intent {event_id} validated successfully. "
                    f"Calculated size: {self._extract_quantity(validated_intent)}"
                )

                symbol = self._extract_symbol(validated_intent) or base_symbol
                quantity = self._extract_quantity(validated_intent)
                price = self._extract_price(validated_intent, portfolio_state)
                confidence = self._extract_confidence(validated_intent)
                if confidence is None:
                    confidence = base_confidence

                notional = abs(float(quantity)) * abs(float(price))
                drift_notional_value = notional

                throttle_snapshot: Mapping[str, Any] | None = None
                throttle_blocked = False
                throttle_decision = self._evaluate_trade_throttle(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    confidence=confidence,
                    notional=notional,
                )
                if throttle_decision is not None and not throttle_decision.allowed:
                    reason = throttle_decision.reason or "trade_throttle_active"
                    throttle_snapshot = throttle_decision.as_dict()
                    human_reason = throttle_snapshot.get("message")
                    metadata_payload: dict[str, Any] = {
                        "reason": reason,
                        "throttle": throttle_snapshot,
                    }
                    if human_reason:
                        metadata_payload["message"] = human_reason
                    if throttle_decision.retry_at is not None:
                        metadata_payload["retry_at"] = throttle_decision.retry_at.astimezone(
                            timezone.utc
                        ).isoformat()
                    if gate_decision_payload is not None:
                        metadata_payload["drift_gate"] = dict(gate_decision_payload)
                    self._record_experiment_event(
                        event_id=event_id,
                        status="throttled",
                        strategy_id=strategy_id,
                        symbol=symbol,
                        confidence=confidence,
                        notional=notional,
                        metadata=metadata_payload,
                        decision=self._get_last_risk_decision(),
                    )
                    log_reason = human_reason or reason
                    logger.warning("Throttled trade intent %s: %s", event_id, log_reason)
                    drift_event_status = "throttled"
                    drift_confidence_value = confidence
                    throttle_blocked = True
                    strategy_stats.record_throttled()
                    trade_outcome = TradeIntentOutcome(
                        status="throttled",
                        executed=False,
                        throttle=throttle_snapshot,
                        metadata=metadata_payload,
                    )
                    if gate_decision is not None and not drift_event_emitted:
                        await self._publish_drift_gate_event(
                            decision=gate_decision,
                            event_id=event_id,
                            status=drift_event_status,
                            strategy_id=strategy_id,
                            symbol=symbol,
                            confidence=drift_confidence_value,
                            notional=drift_notional_value,
                            release_metadata=None,
                        )
                        drift_event_emitted = True

                if not throttle_blocked:
                    cast(Any, self.portfolio_monitor).reserve_position(
                        symbol, float(quantity), price
                    )

                    start = time.perf_counter()
                    current_submitted = coerce_int(
                        self._execution_stats.get("orders_submitted"), default=0
                    )
                    self._execution_stats["orders_submitted"] = current_submitted + 1
                    try:
                        result = await self.execution_engine.process_order(
                            validated_intent
                        )
                    except Exception as exc:
                        strategy_stats.record_failure(str(exc))
                        try:
                            release_quantity = float(quantity)
                        except Exception:
                            release_quantity = coerce_float(quantity, default=0.0)
                        if release_quantity:
                            try:
                                cast(Any, self.portfolio_monitor).release_position(
                                    symbol, release_quantity
                                )
                            except Exception:  # pragma: no cover - diagnostic guardrail
                                logger.debug(
                                    "Failed to release reserved position after execution error",
                                    exc_info=True,
                                )
                        logger.exception(
                            "Execution engine error for trade intent %s", event_id
                        )
                        self._record_execution_failure(exc)
                        decision = self._get_last_risk_decision()
                        failure_metadata: dict[str, object] = {"error": str(exc)}
                        if notional:
                            failure_metadata["notional"] = float(notional)
                        release_metadata = self._extract_release_execution_metadata(
                            validated_intent
                        )
                        if release_metadata:
                            failure_metadata["release_execution"] = release_metadata
                        if release_metadata:
                            await self._publish_release_route_event(
                                event_id=event_id,
                                strategy_id=strategy_id,
                                status="failed",
                                release_metadata=release_metadata,
                            )
                        self._record_experiment_event(
                            event_id=event_id,
                            status="failed",
                            strategy_id=strategy_id,
                            symbol=symbol,
                            confidence=confidence,
                            notional=notional,
                            metadata=failure_metadata,
                            decision=decision,
                        )
                        drift_event_status = "execution_error"
                        drift_confidence_value = confidence
                        drift_release_metadata = release_metadata
                        trade_outcome = TradeIntentOutcome(
                            status="failed",
                            executed=False,
                            throttle=self.get_trade_throttle_snapshot(),
                            metadata=failure_metadata,
                        )
                        if gate_decision is not None and not drift_event_emitted:
                            await self._publish_drift_gate_event(
                                decision=gate_decision,
                                event_id=event_id,
                                status=drift_event_status,
                                strategy_id=strategy_id,
                                symbol=symbol,
                                confidence=drift_confidence_value,
                                notional=drift_notional_value,
                                release_metadata=drift_release_metadata,
                            )
                            drift_event_emitted = True
                    else:
                        latency_ms = (time.perf_counter() - start) * 1000.0
                        self._record_execution_success(latency_ms, result)
                        decision = self._get_last_risk_decision()
                        if self._execution_stats.get("last_successful_order"):
                            side_value = self._extract_side(validated_intent) or side_hint or "BUY"
                            strategy_stats.record_success(
                                symbol=symbol,
                                side=side_value,
                                quantity=float(quantity),
                                price=float(price),
                                notional=float(notional),
                                latency_ms=latency_ms,
                                order_id=result,
                            )
                            if notional > 0:
                                self._roi_executed_trades += 1
                                self._roi_total_notional += notional
                            success_metadata: dict[str, object] = {
                                "latency_ms": float(latency_ms),
                                "notional": float(notional),
                                "quantity": float(quantity),
                                "price": float(price),
                            }
                            if gate_decision_payload is not None:
                                success_metadata["drift_gate"] = dict(gate_decision_payload)
                            release_metadata = self._extract_release_execution_metadata(
                                validated_intent
                            )
                            if release_metadata:
                                success_metadata["release_execution"] = release_metadata
                            if release_metadata:
                                await self._publish_release_route_event(
                                    event_id=event_id,
                                    strategy_id=strategy_id,
                                    status="executed",
                                    release_metadata=release_metadata,
                                )
                            self._record_experiment_event(
                                event_id=event_id,
                                status="executed",
                                strategy_id=strategy_id,
                                symbol=symbol,
                                confidence=confidence,
                                notional=notional,
                                metadata=success_metadata,
                                decision=decision,
                            )
                            drift_event_status = "executed"
                            drift_confidence_value = confidence
                            drift_release_metadata = release_metadata
                            trade_outcome = TradeIntentOutcome(
                                status="executed",
                                executed=True,
                                throttle=self.get_trade_throttle_snapshot(),
                                metadata=success_metadata,
                            )
                            if gate_decision is not None and not drift_event_emitted:
                                await self._publish_drift_gate_event(
                                    decision=gate_decision,
                                    event_id=event_id,
                                    status=drift_event_status,
                                    strategy_id=strategy_id,
                                    symbol=symbol,
                                    confidence=drift_confidence_value,
                                    notional=drift_notional_value,
                                    release_metadata=drift_release_metadata,
                                )
                                drift_event_emitted = True
                        else:
                            error_reason = str(
                                self._execution_stats.get("last_error")
                                or "execution_failed"
                            )
                            strategy_stats.record_failure(error_reason)
                            fallback_metadata: dict[str, object] = {
                                "reason": error_reason
                            }
                            if notional:
                                fallback_metadata["notional"] = float(notional)
                            if gate_decision_payload is not None:
                                fallback_metadata["drift_gate"] = dict(
                                    gate_decision_payload
                                )
                            release_metadata = self._extract_release_execution_metadata(
                                validated_intent
                            )
                            if release_metadata:
                                fallback_metadata["release_execution"] = release_metadata
                            if release_metadata:
                                await self._publish_release_route_event(
                                    event_id=event_id,
                                    strategy_id=strategy_id,
                                    status="failed",
                                    release_metadata=release_metadata,
                                )
                            self._record_experiment_event(
                                event_id=event_id,
                                status="failed",
                                strategy_id=strategy_id,
                                symbol=symbol,
                                confidence=confidence,
                                notional=notional,
                                metadata=fallback_metadata,
                                decision=decision,
                            )
                            drift_event_status = "execution_failed"
                            drift_confidence_value = confidence
                            drift_release_metadata = release_metadata
                            trade_outcome = TradeIntentOutcome(
                                status="failed",
                                executed=False,
                                throttle=self.get_trade_throttle_snapshot(),
                                metadata=fallback_metadata,
                            )
                            if gate_decision is not None and not drift_event_emitted:
                                await self._publish_drift_gate_event(
                                    decision=gate_decision,
                                    event_id=event_id,
                                    status=drift_event_status,
                                    strategy_id=strategy_id,
                                    symbol=symbol,
                                    confidence=drift_confidence_value,
                                    notional=drift_notional_value,
                                    release_metadata=drift_release_metadata,
                                )
                                drift_event_emitted = True

            else:
                logger.warning(f"Trade intent {event_id} was rejected by the Risk Gateway")
                decision = self._get_last_risk_decision()
                reason = None
                if isinstance(decision, Mapping):
                    reason = decision.get("reason")
                strategy_stats.record_rejection(str(reason) if reason else None)
                rejection_metadata: dict[str, object] = {}
                if reason:
                    rejection_metadata["reason"] = reason
                if gate_decision_payload is not None:
                    rejection_metadata["drift_gate"] = dict(gate_decision_payload)
                self._record_experiment_event(
                    event_id=event_id,
                    status="rejected",
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=base_confidence,
                    metadata=rejection_metadata or None,
                    decision=decision,
                )
                drift_event_status = "risk_rejected"
                drift_confidence_value = base_confidence
                trade_outcome = TradeIntentOutcome(
                    status="rejected",
                    executed=False,
                    throttle=self.get_trade_throttle_snapshot(),
                    metadata=rejection_metadata or {},
                )
                if gate_decision is not None and not drift_event_emitted:
                    await self._publish_drift_gate_event(
                        decision=gate_decision,
                        event_id=event_id,
                        status=drift_event_status,
                        strategy_id=strategy_id,
                        symbol=base_symbol,
                        confidence=drift_confidence_value,
                        notional=drift_notional_value,
                        release_metadata=None,
                    )
                    drift_event_emitted = True

        except Exception as e:
            logger.error(f"Error processing trade intent {event_id}: {e}")
            if gate_decision is not None and not drift_event_emitted:
                fallback_status = drift_event_status or "error"
                confidence_payload = (
                    drift_confidence_value
                    if drift_confidence_value is not None
                    else base_confidence
                )
                await self._publish_drift_gate_event(
                    decision=gate_decision,
                    event_id=event_id,
                    status=fallback_status,
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=confidence_payload,
                    notional=drift_notional_value,
                    release_metadata=drift_release_metadata,
                )
                drift_event_emitted = True
            if trade_outcome is None:
                trade_outcome = TradeIntentOutcome(
                    status="error",
                    executed=False,
                    throttle=self.get_trade_throttle_snapshot(),
                    metadata={"error": str(e)},
                )
        finally:
            finished_wall = datetime.now(tz=timezone.utc)
            ingestion_timestamp = self._resolve_intent_timestamp(event)
            lag_ms: float | None = None
            if ingestion_timestamp is not None:
                lag_delta = (started_wall - ingestion_timestamp).total_seconds() * 1000.0
                lag_ms = max(lag_delta, 0.0)
            try:
                self._throughput_monitor.record(
                    started_at=started_wall,
                    finished_at=finished_wall,
                    ingested_at=ingestion_timestamp,
                )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to record throughput metrics", exc_info=True)
            else:
                self._execution_stats["throughput"] = self._throughput_monitor.snapshot()
            backlog_observation = self._backlog_tracker.record(
                lag_ms=lag_ms, timestamp=started_wall
            )
            backlog_snapshot = self._backlog_tracker.snapshot()
            self._execution_stats["backlog"] = backlog_snapshot
            if backlog_observation and backlog_observation.breach:
                breach_count = coerce_int(
                    self._execution_stats.get("backlog_breaches"), default=0
                )
                breach_count += 1
                self._execution_stats["backlog_breaches"] = breach_count
                breach_timestamp = backlog_observation.timestamp.astimezone(
                    timezone.utc
                ).isoformat()
                self._execution_stats["last_backlog_breach"] = breach_timestamp
                metadata_payload: dict[str, Any] = {
                    "lag_ms": float(backlog_observation.lag_ms),
                    "threshold_ms": float(backlog_observation.threshold_ms),
                    "breach_timestamp": breach_timestamp,
                }
                if isinstance(backlog_snapshot, Mapping):
                    metadata_payload["backlog"] = dict(backlog_snapshot)
                logger.warning(
                    "Backlog threshold exceeded for trade intent %s: lag %.2f ms exceeds %.2f ms",
                    event_id,
                    backlog_observation.lag_ms,
                    backlog_observation.threshold_ms,
                )
                self._record_experiment_event(
                    event_id=f"{event_id}-backlog",
                    status="backlog_breach",
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=drift_confidence_value if drift_confidence_value is not None else base_confidence,
                    notional=drift_notional_value,
                    metadata=metadata_payload,
                    decision=self._get_last_risk_decision(),
                )
            resource_snapshot = self._resource_monitor.sample()
            self._execution_stats["resource_usage"] = resource_snapshot
            await self._emit_policy_snapshot()
            await self._emit_risk_interface_snapshot()
            await self._emit_risk_snapshot()

        if trade_outcome is None:
            trade_outcome = TradeIntentOutcome(
                status="noop",
                executed=False,
                throttle=self.get_trade_throttle_snapshot(),
                metadata={},
            )
        return trade_outcome

    def _record_execution_success(self, latency_ms: float, result: object) -> None:
        """Record a successful execution attempt."""

        if result in (False, None, ""):
            self._record_execution_failure("execution_engine_returned_false")
            return

        samples = coerce_int(self._execution_stats.get("latency_samples"), default=0)
        total_latency = coerce_float(
            self._execution_stats.get("total_latency_ms"), default=0.0
        )
        max_latency = coerce_float(self._execution_stats.get("max_latency_ms"), default=0.0)

        samples += 1
        total_latency += float(latency_ms)
        max_latency = max(max_latency, float(latency_ms))

        executed = coerce_int(self._execution_stats.get("orders_executed"), default=0)
        self._execution_stats["orders_executed"] = executed + 1
        self._execution_stats["latency_samples"] = samples
        self._execution_stats["total_latency_ms"] = total_latency
        self._execution_stats["max_latency_ms"] = max_latency
        self._execution_stats["last_execution_at"] = datetime.now(tz=timezone.utc)
        self._execution_stats["last_successful_order"] = result
        self._execution_stats["last_error"] = None

    def _record_execution_failure(self, error: Exception | str) -> None:
        """Record a failed execution attempt."""

        failed = coerce_int(self._execution_stats.get("orders_failed"), default=0)
        self._execution_stats["orders_failed"] = failed + 1
        self._execution_stats["last_error"] = str(error)
        self._execution_stats["last_successful_order"] = None

    def _evaluate_drift_gate(
        self,
        *,
        event: Any,
        strategy_id: str | None,
        symbol: str,
        confidence: float | None,
        portfolio_state: Mapping[str, Any],
        event_id: str,
    ) -> DriftSentryDecision | None:
        if self._drift_gate is None:
            return None

        quantity = self._extract_quantity(event)
        try:
            quantity_value = float(quantity)
        except Exception:
            quantity_value = coerce_float(quantity, default=0.0) or 0.0
        quantity_abs = abs(quantity_value)

        price = self._extract_price(event, portfolio_state)
        notional = abs(quantity_abs * float(price)) if price is not None else None

        context_metadata: dict[str, Any] = {"event_id": event_id, "symbol": symbol}
        if quantity_abs:
            context_metadata["quantity"] = quantity_abs
        if notional is not None:
            context_metadata["notional"] = notional

        threshold_payload: Mapping[str, Any] | None = None
        snapshot = self._drift_gate.latest_snapshot
        if self._adaptive_thresholds is not None:
            try:
                threshold_payload = self._adaptive_thresholds.resolve(
                    strategy_id=strategy_id,
                    snapshot=snapshot,
                )
            except Exception:  # pragma: no cover - defensive guard
                threshold_payload = None
        elif self._release_manager is not None:
            try:
                threshold_payload = self._release_manager.resolve_thresholds(strategy_id)
            except Exception:  # pragma: no cover - defensive guard
                threshold_payload = None
        if threshold_payload:
            stage_value = str(threshold_payload.get("stage") or "").strip()
            if stage_value:
                context_metadata.setdefault("release_stage", stage_value)

        decision = self._drift_gate.evaluate_trade(
            symbol=symbol,
            strategy_id=strategy_id,
            confidence=confidence,
            quantity=quantity_abs if quantity_abs else None,
            notional=notional,
            metadata=context_metadata,
            threshold_overrides=threshold_payload,
        )
        self._last_drift_gate_decision = decision
        return decision

    def _record_experiment_event(
        self,
        *,
        event_id: str,
        status: str,
        strategy_id: str | None,
        symbol: str | None,
        confidence: float | None,
        notional: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "event_id": event_id,
            "status": status,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        if strategy_id:
            entry["strategy_id"] = strategy_id
        if symbol:
            entry["symbol"] = symbol
        if confidence is not None:
            entry["confidence"] = confidence
        if notional is not None:
            entry["notional"] = float(notional)
        if metadata and isinstance(metadata, Mapping):
            cleaned = {
                key: value for key, value in dict(metadata).items() if value not in (None, "")
            }
            if cleaned:
                entry["metadata"] = cleaned
        if decision and isinstance(decision, Mapping):
            entry["decision"] = dict(decision)
        self._experiment_events.appendleft(entry)

    async def _publish_drift_gate_event(
        self,
        *,
        decision: DriftSentryDecision,
        event_id: str,
        status: str,
        strategy_id: str | None,
        symbol: str | None,
        confidence: float | None,
        notional: float | None,
        release_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish drift gate telemetry without interrupting trade processing."""

        try:
            event = DriftGateEvent(
                event_id=event_id,
                strategy_id=strategy_id,
                symbol=symbol,
                status=status,
                decision=decision,
                confidence=confidence,
                notional=notional,
                release=release_metadata,
            )
        except Exception:  # pragma: no cover - telemetry guardrail
            logger.debug(
                "Failed to assemble drift gate telemetry payload",
                exc_info=True,
            )
            return

        try:
            await publish_drift_gate_event(
                self.event_bus,
                event,
                source="trading_manager",
            )
        except Exception:  # pragma: no cover - telemetry guardrail
            logger.debug(
                "Failed to publish drift gate telemetry event",
                exc_info=True,
            )

    def _attach_drift_gate_metadata(
        self,
        intent: Any,
        gate_payload: Mapping[str, Any] | None,
    ) -> None:
        """Inject DriftSentry metadata into an intent for downstream consumers."""

        if intent is None or not gate_payload:
            return

        try:
            payload = dict(gate_payload)
        except Exception:
            return

        if isinstance(intent, MutableMappingABC):
            metadata = intent.get("metadata")
            if isinstance(metadata, MutableMappingABC):
                metadata["drift_gate"] = payload
            else:
                intent["metadata"] = {"drift_gate": payload}
            return

        metadata_attr = getattr(intent, "metadata", None)
        if isinstance(metadata_attr, MutableMappingABC):
            metadata_attr["drift_gate"] = payload
            return

        if metadata_attr is None:
            setattr(intent, "metadata", {"drift_gate": payload})
            return

        if isinstance(metadata_attr, MappingABC):
            metadata_dict = dict(metadata_attr)
        else:
            try:
                metadata_dict = dict(metadata_attr)  # type: ignore[arg-type]
            except Exception:
                metadata_dict = {"original_metadata": metadata_attr}

        metadata_dict["drift_gate"] = payload
        setattr(intent, "metadata", metadata_dict)

    async def _publish_release_route_event(
        self,
        *,
        event_id: str,
        strategy_id: str | None,
        status: str,
        release_metadata: Mapping[str, Any] | None,
    ) -> None:
        """Publish release routing telemetry without impacting trade processing."""

        if not release_metadata:
            return

        try:
            forced_reasons_candidate = release_metadata.get("forced_reasons")
            forced_reasons: tuple[str, ...]
            if isinstance(forced_reasons_candidate, (list, tuple)):
                forced_reasons = tuple(
                    str(reason) for reason in forced_reasons_candidate if reason
                )
            else:
                forced_reasons = ()
            overridden_flag = release_metadata.get("overridden")
            audit_payload = release_metadata.get("audit")
            audit_mapping = (
                dict(audit_payload)
                if isinstance(audit_payload, Mapping)
                else None
            )
            drift_severity_value = release_metadata.get("drift_severity")
            event = ReleaseRouteEvent(
                event_id=event_id,
                strategy_id=strategy_id,
                status=status,
                stage=str(release_metadata.get("stage"))
                if release_metadata.get("stage")
                else None,
                route=str(release_metadata.get("route"))
                if release_metadata.get("route")
                else None,
                forced=bool(release_metadata.get("forced")),
                forced_reason=(
                    str(release_metadata.get("forced_reason"))
                    if release_metadata.get("forced_reason")
                    else None
                ),
                forced_reasons=forced_reasons,
                overridden=overridden_flag if isinstance(overridden_flag, bool) else None,
                audit=audit_mapping,
                drift_severity=str(drift_severity_value)
                if drift_severity_value
                else None,
                metadata=dict(release_metadata),
            )
        except Exception:  # pragma: no cover - telemetry guardrail
            logger.debug(
                "Failed to assemble release route telemetry payload",
                exc_info=True,
            )
            return

        try:
            await publish_release_route_event(
                self.event_bus,
                event,
                source="trading_manager",
            )
        except Exception:  # pragma: no cover - telemetry guardrail
            logger.debug(
                "Failed to publish release route telemetry event",
                exc_info=True,
            )

    def _extract_release_execution_metadata(self, intent: Any) -> Mapping[str, Any] | None:
        """Extract release routing details exposed by the execution router."""

        metadata_candidate: Mapping[str, Any] | None
        if isinstance(intent, MutableMappingABC):
            raw = intent.get("metadata")
            metadata_candidate = raw if isinstance(raw, Mapping) else None
        else:
            raw_attr = getattr(intent, "metadata", None)
            metadata_candidate = raw_attr if isinstance(raw_attr, Mapping) else None

        if not metadata_candidate:
            return None

        payload: dict[str, Any] = {}

        stage = metadata_candidate.get("release_stage")
        if stage:
            payload["stage"] = str(stage)

        route = metadata_candidate.get("release_execution_route")
        if route:
            payload["route"] = str(route)

        forced_reason = metadata_candidate.get("release_execution_forced")
        if forced_reason:
            payload["forced_reason"] = str(forced_reason)

        overridden_flag = metadata_candidate.get("release_execution_route_overridden")
        overridden = overridden_flag if isinstance(overridden_flag, bool) else None
        if overridden is not None:
            payload["overridden"] = overridden

        forced_reasons_candidate = metadata_candidate.get("release_execution_forced_reasons")
        forced_reasons: list[str] = []
        if isinstance(forced_reasons_candidate, (list, tuple)):
            forced_reasons = [str(reason) for reason in forced_reasons_candidate if reason]
            if forced_reasons:
                payload["forced_reasons"] = forced_reasons

        audit_payload = metadata_candidate.get("release_execution_audit")
        if isinstance(audit_payload, Mapping):
            payload["audit"] = dict(audit_payload)

        if forced_reason or overridden or forced_reasons:
            payload["forced"] = True

        drift_gate_payload = metadata_candidate.get("drift_gate")
        if isinstance(drift_gate_payload, Mapping):
            severity = drift_gate_payload.get("severity")
            if severity:
                payload.setdefault("drift_severity", str(severity))

        return payload or None

    def _get_last_risk_decision(self) -> Mapping[str, Any] | None:
        """Safely fetch the last risk decision from the gateway."""

        try:
            return cast(Optional[Mapping[str, Any]], self.risk_gateway.get_last_decision())
        except Exception:
            return None

    @staticmethod
    def _coerce_event_timestamp(candidate: Any) -> datetime | None:
        """Convert common timestamp representations to ``datetime``."""

        if candidate is None:
            return None
        if isinstance(candidate, datetime):
            if candidate.tzinfo is None:
                return candidate.replace(tzinfo=timezone.utc)
            return candidate
        if isinstance(candidate, (int, float)):
            try:
                return datetime.fromtimestamp(float(candidate), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    @classmethod
    def _resolve_intent_timestamp(cls, event: TradeIntent) -> datetime | None:
        """Resolve the best available timestamp for when an intent entered the loop."""

        for attr in ("ingested_at", "created_at", "timestamp", "ts"):
            candidate = getattr(event, attr, None)
            resolved = cls._coerce_event_timestamp(candidate)
            if resolved is not None:
                return resolved
        return None

    def get_execution_stats(self) -> Mapping[str, object]:
        """Return execution telemetry derived from the configured engine."""

        stats: dict[str, object] = dict(self._execution_stats)
        samples = coerce_int(stats.get("latency_samples"), default=0)
        total_latency = coerce_float(stats.get("total_latency_ms"), default=0.0)
        stats["avg_latency_ms"] = total_latency / samples if samples > 0 else None

        engine = getattr(self, "execution_engine", None)
        pending_orders = None
        fills = None
        if engine is not None:
            try:
                if hasattr(engine, "iter_orders"):
                    pending_orders = sum(
                        1
                        for order in engine.iter_orders()
                        if getattr(order, "status", None) not in {"FILLED", "CANCELLED"}
                    )
            except Exception:  # pragma: no cover - diagnostics only
                pending_orders = None
            try:
                active_orders = getattr(engine, "active_orders", None)
                if pending_orders is None and isinstance(active_orders, MappingABC):
                    pending_orders = len(active_orders)
            except Exception:  # pragma: no cover - diagnostics only
                pending_orders = pending_orders
            try:
                fills_attr = getattr(engine, "fills", None)
                if isinstance(fills_attr, list):
                    fills = len(fills_attr)
            except Exception:  # pragma: no cover - diagnostics only
                fills = None

        if pending_orders is not None:
            stats["pending_orders"] = pending_orders
        if fills is not None:
            stats["fills"] = fills

        stats.setdefault("throughput", self._throughput_monitor.snapshot())

        strategy_summary = self.get_strategy_execution_summary()
        if strategy_summary:
            stats["strategies"] = strategy_summary

        return stats

    def generate_execution_report(self) -> str:
        """Render a Markdown summary of execution throughput and throttle posture."""

        return build_execution_performance_report(self.get_execution_stats())

    def generate_performance_health_report(
        self,
        *,
        max_processing_ms: float = 250.0,
        max_lag_ms: float = 250.0,
        backlog_threshold_ms: float | None = None,
        max_cpu_percent: float | None = None,
        max_memory_mb: float | None = None,
        max_memory_percent: float | None = None,
    ) -> str:
        """Render a Markdown summary of the current performance health posture."""

        assessment = self.assess_performance_health(
            max_processing_ms=max_processing_ms,
            max_lag_ms=max_lag_ms,
            backlog_threshold_ms=backlog_threshold_ms,
            max_cpu_percent=max_cpu_percent,
            max_memory_mb=max_memory_mb,
            max_memory_percent=max_memory_percent,
        )
        return build_performance_health_report(assessment)

    def assess_throughput_health(
        self,
        *,
        max_processing_ms: float = 250.0,
        max_lag_ms: float = 250.0,
    ) -> Mapping[str, object]:
        """Evaluate whether recent throughput metrics stay within limits."""

        stats = self.get_execution_stats()
        throughput_stats = stats.get("throughput")

        if not isinstance(throughput_stats, MappingABC):
            return {
                "healthy": False,
                "processing_within_limit": False,
                "lag_within_limit": False,
                "max_processing_ms": None,
                "max_lag_ms": None,
                "samples": 0,
            }

        samples = coerce_int(throughput_stats.get("samples"), default=0) or 0
        max_processing = coerce_float(
            throughput_stats.get("max_processing_ms"), default=None
        )
        max_lag = coerce_float(throughput_stats.get("max_lag_ms"), default=None)

        processing_within_limit = (
            max_processing is None or max_processing <= float(max_processing_ms)
        )
        lag_within_limit = max_lag is None or max_lag <= float(max_lag_ms)

        return {
            "healthy": processing_within_limit and lag_within_limit,
            "processing_within_limit": processing_within_limit,
            "lag_within_limit": lag_within_limit,
            "max_processing_ms": max_processing,
            "max_lag_ms": max_lag,
            "samples": samples,
        }

    def assess_performance_health(
        self,
        *,
        max_processing_ms: float = 250.0,
        max_lag_ms: float = 250.0,
        backlog_threshold_ms: float | None = None,
        max_cpu_percent: float | None = None,
        max_memory_mb: float | None = None,
        max_memory_percent: float | None = None,
    ) -> Mapping[str, object]:
        """Evaluate latency, backlog, and resource posture against budgets."""

        throughput_health = dict(
            self.assess_throughput_health(
                max_processing_ms=max_processing_ms, max_lag_ms=max_lag_ms
            )
        )

        backlog_snapshot_raw = self._backlog_tracker.snapshot()
        backlog_snapshot = (
            dict(backlog_snapshot_raw) if isinstance(backlog_snapshot_raw, MappingABC) else {}
        )
        backlog_samples = coerce_int(backlog_snapshot.get("samples"), default=0)
        if backlog_threshold_ms is not None:
            resolved_backlog_threshold = float(backlog_threshold_ms)
        else:
            snapshot_threshold = backlog_snapshot.get("threshold_ms")
            if snapshot_threshold is not None:
                resolved_backlog_threshold = float(snapshot_threshold)
            else:
                resolved_backlog_threshold = float(getattr(self._backlog_tracker, "threshold_ms", 0.0))

        backlog_max_lag = backlog_snapshot.get("max_lag_ms")
        backlog_max_lag_value = float(backlog_max_lag) if backlog_max_lag is not None else None
        backlog_evaluated = backlog_samples > 0 and backlog_max_lag_value is not None
        if backlog_evaluated:
            backlog_healthy = backlog_max_lag_value <= resolved_backlog_threshold
        else:
            backlog_healthy = bool(backlog_snapshot.get("healthy", True))

        backlog_result: dict[str, object] = {
            "healthy": backlog_healthy,
            "evaluated": backlog_evaluated,
            "threshold_ms": resolved_backlog_threshold,
            "samples": backlog_samples,
            "max_lag_ms": backlog_max_lag_value,
            "snapshot": backlog_snapshot,
        }

        resource_snapshot_raw = self._resource_monitor.snapshot()
        resource_snapshot = (
            dict(resource_snapshot_raw)
            if isinstance(resource_snapshot_raw, MappingABC)
            else {}
        )
        resource_limits = {
            "max_cpu_percent": float(max_cpu_percent) if max_cpu_percent is not None else None,
            "max_memory_mb": float(max_memory_mb) if max_memory_mb is not None else None,
            "max_memory_percent": float(max_memory_percent)
            if max_memory_percent is not None
            else None,
        }
        limit_to_field = {
            "max_cpu_percent": "cpu_percent",
            "max_memory_mb": "memory_mb",
            "max_memory_percent": "memory_percent",
        }
        limits_configured = any(value is not None for value in resource_limits.values())
        resource_status = "not_configured"
        resource_healthy = True
        violations: dict[str, dict[str, float]] = {}
        evaluated_metrics: list[str] = []
        if limits_configured:
            resource_status = "no_data"
            resource_healthy = False
            for limit_key, limit_value in resource_limits.items():
                if limit_value is None:
                    continue
                field = limit_to_field[limit_key]
                sample_value = resource_snapshot.get(field)
                if sample_value is None:
                    continue
                evaluated_metrics.append(limit_key)
                try:
                    numeric_value = float(sample_value)
                except (TypeError, ValueError):
                    continue
                if numeric_value <= float(limit_value):
                    continue
                resource_status = "violated"
                resource_healthy = False
                violations[limit_key] = {
                    "value": numeric_value,
                    "limit": float(limit_value),
                }
            if evaluated_metrics and not violations:
                resource_status = "ok"
                resource_healthy = True
            elif not evaluated_metrics:
                resource_status = "no_data"
                resource_healthy = False

        resource_result: dict[str, object] = {
            "healthy": resource_healthy,
            "status": resource_status,
            "limits": resource_limits,
            "evaluated_metrics": evaluated_metrics,
            "violations": violations,
            "sample": resource_snapshot,
        }

        throttle_summary: Mapping[str, Any] | None = None
        throttle_snapshot = self.get_trade_throttle_snapshot()
        if isinstance(throttle_snapshot, MappingABC):
            summary: dict[str, Any] = {
                "state": throttle_snapshot.get("state"),
                "active": bool(throttle_snapshot.get("active", False)),
                "reason": throttle_snapshot.get("reason"),
                "message": throttle_snapshot.get("message"),
            }
            metadata = throttle_snapshot.get("metadata")
            if isinstance(metadata, MappingABC):
                retry_at = metadata.get("retry_at")
                if retry_at is not None:
                    summary["retry_at"] = retry_at
                context = metadata.get("context")
                if isinstance(context, MappingABC) and context:
                    summary["context"] = dict(context)
            throttle_summary = summary

        overall_checks: list[bool] = [bool(throughput_health.get("healthy"))]
        if backlog_result["evaluated"]:
            overall_checks.append(bool(backlog_result["healthy"]))
        if resource_status == "violated":
            overall_checks.append(False)
        elif resource_status == "ok":
            overall_checks.append(True)
        elif resource_status == "no_data":
            overall_checks.append(False)

        overall_healthy = all(overall_checks) if overall_checks else False

        result: dict[str, object] = {
            "healthy": overall_healthy,
            "throughput": throughput_health,
            "backlog": backlog_result,
            "resource": resource_result,
        }
        if throttle_summary is not None:
            result["throttle"] = throttle_summary
        return result

    def get_trade_throttle_snapshot(self) -> Mapping[str, Any] | None:
        """Expose the most recent trade throttle posture."""

        if self._trade_throttle_snapshot is None:
            return None
        snapshot = dict(self._trade_throttle_snapshot)
        metadata = snapshot.get("metadata")
        if isinstance(metadata, Mapping):
            snapshot["metadata"] = dict(metadata)
        return snapshot

    def get_experiment_events(self, limit: int | None = None) -> list[Mapping[str, Any]]:
        """Expose the recent paper-trading experiment events."""

        events: list[Mapping[str, Any]] = [dict(event) for event in self._experiment_events]
        if limit is None:
            return events
        count = int(limit)
        if count <= 0:
            return []
        return events[: min(len(events), count)]

    def attach_drift_gate(self, gate: DriftSentryGate | None) -> None:
        """Attach or replace the configured DriftSentry gate."""

        self._drift_gate = gate
        self._last_drift_gate_decision = None

    def update_drift_sentry_snapshot(self, snapshot: SensoryDriftSnapshot | None) -> None:
        """Propagate the latest sensory drift snapshot to the gate, if configured."""

        if self._drift_gate is None:
            return
        self._drift_gate.update_snapshot(snapshot)

    def get_last_drift_gate_decision(self) -> DriftSentryDecision | None:
        """Expose the most recent DriftSentry decision for observability surfaces."""

        return self._last_drift_gate_decision

    def get_last_release_route(self) -> Mapping[str, Any] | None:
        """Expose the most recent release-aware execution routing decision."""

        if self._release_router is None:
            return None
        try:
            last_route = self._release_router.last_route()
        except Exception:  # pragma: no cover - diagnostic fallback
            logger.debug(
                "Failed to fetch last release route from execution router",
                exc_info=True,
            )
            return None
        if not last_route:
            return None
        return dict(last_route)

    def describe_drift_gate(self) -> Mapping[str, Any]:
        """Return a serialisable posture for the DriftSentry gate."""

        if self._drift_gate is None:
            return {"enabled": False}
        description = dict(self._drift_gate.describe())
        description["enabled"] = True
        return description

    def describe_release_posture(self, strategy_id: str | None = None) -> Mapping[str, Any]:
        """Expose the release posture derived from the policy ledger."""

        if self._release_manager is None:
            payload: dict[str, Any] = {
                "managed": False,
                "stage": PolicyLedgerStage.EXPERIMENT.value,
                "thresholds": {},
            }
            if strategy_id:
                payload["strategy_id"] = strategy_id
            return payload

        try:
            summary = dict(self._release_manager.describe(strategy_id))
        except Exception:  # pragma: no cover - defensive guard
            stage = self._release_manager.resolve_stage(strategy_id)
            try:
                thresholds = dict(self._release_manager.resolve_thresholds(strategy_id))
            except Exception:
                thresholds = {}
            summary = {"stage": stage.value, "thresholds": thresholds}
        else:
            if self._adaptive_thresholds is not None:
                snapshot = self._drift_gate.latest_snapshot if self._drift_gate else None
                try:
                    adaptive_thresholds = self._adaptive_thresholds.resolve(
                        strategy_id=strategy_id,
                        snapshot=snapshot,
                    )
                except Exception:  # pragma: no cover - defensive guard
                    adaptive_thresholds = None
                if adaptive_thresholds:
                    summary["thresholds"] = dict(adaptive_thresholds)
            last_route = self.get_last_release_route()
            if last_route:
                summary["last_route"] = last_route
        summary.setdefault("managed", True)
        if strategy_id and "strategy_id" not in summary:
            summary["strategy_id"] = strategy_id
        return summary

    def build_policy_governance_snapshot(
        self,
        *,
        regulation: str = "AlphaTrade Governance",
        generated_at: datetime | None = None,
    ) -> ComplianceWorkflowSnapshot | None:
        """Return the governance checklist derived from the policy ledger."""

        if self._release_manager is None:
            return None
        try:
            return self._release_manager.build_governance_workflow(
                regulation=regulation,
                generated_at=generated_at,
            )
        except Exception:  # pragma: no cover - diagnostic fallback
            logger.debug(
                "Failed to build policy governance workflow snapshot",
                exc_info=True,
            )
            return None

    def install_release_execution_router(
        self,
        *,
        paper_engine: Any | None = None,
        pilot_engine: Any | None = None,
        live_engine: Any | None = None,
        default_stage: PolicyLedgerStage | str | None = None,
    ) -> ReleaseAwareExecutionRouter:
        """Wrap execution with release-aware routing tied to the policy ledger."""

        if self._release_manager is None:
            raise RuntimeError("Release manager is not configured")

        if isinstance(self.execution_engine, ReleaseAwareExecutionRouter):
            base_paper = self.execution_engine.paper_engine
        else:
            base_paper = paper_engine or self.execution_engine

        if base_paper is None:
            raise ValueError("paper_engine must be provided when no base engine is configured")

        stage_default = (
            PolicyLedgerStage.from_value(default_stage)
            if default_stage is not None
            else PolicyLedgerStage.EXPERIMENT
        )

        if pilot_engine is not None:
            self._pilot_engine = pilot_engine
        if live_engine is not None:
            self._live_engine = live_engine

        router = ReleaseAwareExecutionRouter(
            release_manager=self._release_manager,
            paper_engine=base_paper,
            pilot_engine=self._pilot_engine,
            live_engine=self._live_engine,
            default_stage=stage_default,
        )
        self.execution_engine = router
        self._release_router = router
        self._configure_execution_risk_context(router)
        return router

    def configure_release_execution(
        self,
        *,
        pilot_engine: Any | object = _ENGINE_UNSET,
        live_engine: Any | object = _ENGINE_UNSET,
    ) -> ReleaseAwareExecutionRouter | None:
        """Configure the engines used for pilot and limited-live release stages."""

        if pilot_engine is not _ENGINE_UNSET:
            self._pilot_engine = pilot_engine
        if live_engine is not _ENGINE_UNSET:
            self._live_engine = live_engine

        router = self._release_router

        if router is None:
            if self._release_manager is None:
                return None
            base_engine = self.execution_engine
            if base_engine is None:
                return None
            return self.install_release_execution_router(
                paper_engine=base_engine,
                pilot_engine=self._pilot_engine,
                live_engine=self._live_engine,
            )

        kwargs: dict[str, Any] = {}
        if pilot_engine is not _ENGINE_UNSET:
            kwargs["pilot_engine"] = self._pilot_engine
        if live_engine is not _ENGINE_UNSET:
            kwargs["live_engine"] = self._live_engine
        if kwargs:
            router.configure_engines(**kwargs)
        return router

    def attach_live_broker_adapter(
        self,
        broker_interface: Any,
        *,
        default_stage: PolicyLedgerStage | str | None = None,
        order_timeout: float | None = 5.0,
    ) -> ReleaseAwareExecutionRouter | None:
        """Install a paper broker adapter as the live execution engine.

        This bridges validated intents into the FIX paper trading stack when the
        governance ledger promotes a tactic into the ``limited_live`` stage.
        """

        if broker_interface is None:
            raise ValueError("broker_interface must be provided")

        adapter = PaperBrokerExecutionAdapter(
            broker_interface=broker_interface,
            portfolio_monitor=self.portfolio_monitor,
            order_timeout=order_timeout,
        )

        if self._release_manager is None:
            logger.warning(
                "Release manager is not configured; skipping paper broker adapter installation",
            )
            self._live_engine = adapter
            return None

        stage_default: PolicyLedgerStage | None = None
        if default_stage is not None:
            try:
                stage_default = PolicyLedgerStage.from_value(default_stage)
            except ValueError as exc:
                raise ValueError(f"Unknown policy ledger stage: {default_stage}") from exc

        self._live_engine = adapter

        router = self.configure_release_execution(live_engine=adapter)
        if router is None:
            base_engine = self.execution_engine
            if base_engine is None:
                raise RuntimeError(
                    "TradingManager must have a base execution engine before attaching the paper broker adapter",
                )
            resolved_default = stage_default
            if resolved_default is None:
                try:
                    resolved_default = self._release_manager.resolve_stage(None)
                except Exception:
                    resolved_default = PolicyLedgerStage.EXPERIMENT

            router = self.install_release_execution_router(
                paper_engine=base_engine,
                pilot_engine=self._pilot_engine or base_engine,
                live_engine=adapter,
                default_stage=resolved_default,
            )
        elif stage_default is not None and router.default_stage is not stage_default:
            router.default_stage = stage_default

        return router

    def _maybe_auto_install_release_router(self) -> None:
        """Automatically enable release-aware routing when possible."""

        if self._release_manager is None:
            return

        base_engine = self.execution_engine
        if base_engine is None:
            return

        if isinstance(base_engine, ReleaseAwareExecutionRouter):
            self._release_router = base_engine
            return

        try:
            default_stage: PolicyLedgerStage | None
            try:
                default_stage = self._release_manager.resolve_stage(None)
            except Exception:
                default_stage = None

            self.install_release_execution_router(
                paper_engine=base_engine,
                pilot_engine=self._pilot_engine or base_engine,
                live_engine=self._live_engine or base_engine,
                default_stage=default_stage,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Failed to auto-install release-aware execution router",
                exc_info=True,
            )

    def _strategy_stats_for(self, strategy_id: str | None) -> StrategyExecutionStats:
        key = strategy_id or "unknown"
        stats = self._strategy_stats.get(key)
        if stats is None:
            stats = StrategyExecutionStats(strategy_id=key)
            self._strategy_stats[key] = stats
        return stats

    def get_strategy_execution_summary(self) -> Mapping[str, Any]:
        """Expose aggregated execution telemetry per strategy."""

        return {
            strategy_id: stats.as_dict()
            for strategy_id, stats in sorted(self._strategy_stats.items())
        }

    def describe_release_execution(self) -> Mapping[str, Any] | None:
        """Expose the configured release-aware execution routing summary."""

        if self._release_router is None:
            return None
        try:
            return self._release_router.describe()
        except Exception:  # pragma: no cover - diagnostic fallback
            logger.debug(
                "Failed to describe release execution router",
                exc_info=True,
            )
            return None

    async def start(self) -> None:
        """Start the TradingManager and subscribe to trade intents."""
        logger.info("Starting TradingManager...")
        self._roi_period_start = datetime.now(tz=timezone.utc)
        # In real implementation, would subscribe to event bus
        # await self.event_bus.subscribe("trade_intent", self.on_trade_intent)

    async def stop(self) -> None:
        """Stop the TradingManager."""
        logger.info("Stopping TradingManager...")

    def get_risk_status(self) -> dict[str, object]:
        """
        Get current risk management status.

        Returns:
            Dictionary with current risk configuration and portfolio state
        """
        gateway = cast(Any, self.risk_gateway)
        limits_payload = self._resolve_gateway_limits()

        payload: dict[str, object] = {
            "risk_limits": dict(limits_payload) if limits_payload is not None else {},
            "portfolio_state": cast(Any, self.portfolio_monitor).get_state(),
        }

        gateway_reference: Mapping[str, object] | None = None
        if isinstance(limits_payload, Mapping):
            runbook_candidate = limits_payload.get("runbook")
            if isinstance(runbook_candidate, str) and runbook_candidate:
                payload["risk_api_runbook"] = runbook_candidate
            reference_candidate = limits_payload.get("risk_reference")
            if isinstance(reference_candidate, Mapping):
                gateway_reference = reference_candidate
                payload["risk_reference"] = dict(reference_candidate)

        last_decision = gateway.get_last_decision()
        if isinstance(last_decision, Mapping):
            risk_manager_info = last_decision.get("risk_manager")
            if isinstance(risk_manager_info, Mapping):
                payload["risk_manager"] = dict(risk_manager_info)
        if self._last_policy_snapshot is not None:
            payload["policy"] = {
                "snapshot": self._last_policy_snapshot.as_dict(),
                "markdown": format_policy_markdown(self._last_policy_snapshot),
            }
        if hasattr(self, "_risk_policy"):
            payload["policy_limits"] = self._risk_policy.limit_snapshot()
            payload["policy_research_mode"] = self._risk_policy.research_mode
        config = getattr(self, "_risk_config", None)
        if isinstance(config, TradingRiskConfig):
            try:
                payload["risk_config"] = config.dict()
            except Exception:
                payload["risk_config"] = {}
            else:
                try:
                    summary = summarise_risk_config(config)
                except Exception:
                    logger.debug("Failed to summarise trading risk config", exc_info=True)
                else:
                    payload["risk_config_summary"] = summary
        else:
            payload["risk_config"] = {}

        resolved_summary, metadata_error = self._resolve_risk_metadata()
        if resolved_summary is not None:
            summary_copy = dict(resolved_summary)
            payload["risk_config_summary"] = summary_copy
            runbook = summary_copy.get("runbook")
            if isinstance(runbook, str) and runbook:
                payload["risk_api_runbook"] = runbook
            reference_update: dict[str, object] = {
                "risk_api_runbook": payload.get("risk_api_runbook", RISK_API_RUNBOOK),
                "risk_config_summary": {
                    key: value for key, value in summary_copy.items() if key != "runbook"
                },
            }
            existing_reference = payload.get("risk_reference")
            base_reference = (
                existing_reference if isinstance(existing_reference, Mapping) else gateway_reference
            )
            payload["risk_reference"] = self._merge_risk_reference(
                base_reference,
                reference_update,
            )
        elif metadata_error is not None:
            payload["risk_interface_error"] = metadata_error
            runbook = metadata_error.get("runbook")
            if isinstance(runbook, str) and runbook:
                payload["risk_api_runbook"] = runbook

        payload.setdefault("risk_api_runbook", RISK_API_RUNBOOK)
        if "risk_reference" not in payload:
            payload["risk_reference"] = {
                "risk_api_runbook": payload["risk_api_runbook"],
            }
        if self._last_risk_snapshot is not None:
            payload["snapshot"] = self._last_risk_snapshot.as_dict()
        return payload

    def get_last_risk_snapshot(self) -> RiskTelemetrySnapshot | None:
        """Return the last emitted risk telemetry snapshot, if available."""

        return self._last_risk_snapshot

    def get_last_roi_snapshot(self) -> RoiTelemetrySnapshot | None:
        """Return the last emitted ROI telemetry snapshot, if available."""

        return self._last_roi_snapshot

    def get_last_policy_snapshot(self) -> RiskPolicyEvaluationSnapshot | None:
        """Expose the last published risk policy decision snapshot."""

        return self._last_policy_snapshot

    def get_last_risk_interface_snapshot(self) -> RiskInterfaceSnapshot | None:
        """Expose the last published trading risk interface snapshot."""

        return self._last_risk_interface_snapshot

    def get_last_risk_interface_error(self) -> RiskInterfaceErrorAlert | None:
        """Expose the last risk interface enforcement error, if any."""

        return self._last_risk_interface_error

    def describe_risk_interface(self) -> dict[str, object]:
        """Expose a deterministic snapshot of the trading risk interface."""

        try:
            interface = resolve_trading_risk_interface(self)
        except RiskApiError as exc:
            return {
                "error": str(exc),
                "runbook": exc.runbook,
                "details": exc.to_metadata().get("details", {}),
            }

        summary = interface.summary()
        payload: dict[str, object] = {
            "config": interface.config.dict(),
            "summary": summary,
        }
        if interface.status is not None:
            payload["status"] = dict(interface.status)

        runbook = summary.get("runbook") if isinstance(summary, Mapping) else None
        if isinstance(runbook, str) and runbook:
            payload["runbook"] = runbook

        gateway_reference = self._extract_gateway_reference()
        summary_reference: dict[str, object] = {
            "risk_api_runbook": payload.get("runbook", RISK_API_RUNBOOK),
            "risk_config_summary": {
                key: value for key, value in summary.items() if key != "runbook"
            }
            if isinstance(summary, Mapping)
            else {},
        }
        payload["risk_reference"] = self._merge_risk_reference(
            gateway_reference,
            summary_reference,
        )
        payload.setdefault("runbook", payload["risk_reference"].get("risk_api_runbook", RISK_API_RUNBOOK))
        return payload

    def _resolve_redis_client(self, provided: Any | None) -> Any:
        if provided is not None:
            return provided

        if redis is not None:
            try:
                client = redis.Redis(host="localhost", port=6379, db=0)
                client.ping()
                return client
            except Exception:
                logger.warning(
                    "Redis connection unavailable, reverting to in-memory portfolio store"
                )
        return InMemoryRedis()

    def _configure_liquidity_prober(
        self,
        prober: Any | None,
        task_supervisor: "TaskSupervisor | None",
    ) -> Any | None:
        if prober is None:
            return None

        if task_supervisor is not None:
            setter = getattr(prober, "set_task_supervisor", None)
            if callable(setter):
                try:
                    setter(task_supervisor)
                except Exception:  # pragma: no cover - diagnostic guardrail
                    logger.debug(
                        "Failed to attach task supervisor to liquidity prober",
                        exc_info=True,
                    )
            elif hasattr(prober, "_task_supervisor"):
                try:
                    setattr(prober, "_task_supervisor", task_supervisor)
                except Exception:  # pragma: no cover - diagnostic guardrail
                    logger.debug(
                        "Failed to set liquidity prober task supervisor attribute",
                        exc_info=True,
                    )

        provider: Callable[[], Any] = lambda: self
        provider_setter = getattr(prober, "set_risk_context_provider", None)
        if callable(provider_setter):
            try:
                provider_setter(provider)
            except Exception:  # pragma: no cover - diagnostic guardrail
                logger.debug(
                    "Failed to attach risk context provider to liquidity prober",
                    exc_info=True,
                )
        elif hasattr(prober, "_risk_context_provider"):
            try:
                setattr(prober, "_risk_context_provider", provider)
            except Exception:  # pragma: no cover - diagnostic guardrail
                logger.debug(
                    "Failed to set liquidity prober risk context attribute",
                    exc_info=True,
                )
        return prober

    def attach_task_supervisor(self, task_supervisor: "TaskSupervisor | None") -> None:
        """Bind a task supervisor after initialisation for probe supervision."""

        self._task_supervisor = task_supervisor
        if self.liquidity_prober is not None:
            self.liquidity_prober = self._configure_liquidity_prober(
                self.liquidity_prober,
                task_supervisor,
            )

    def _resolve_gateway_limits(self) -> Mapping[str, object] | None:
        try:
            payload = cast(Any, self.risk_gateway).get_risk_limits()
        except Exception:
            logger.debug("Failed to resolve risk gateway limits", exc_info=True)
            return None
        if isinstance(payload, Mapping):
            return payload
        return None

    def _configure_execution_risk_context(self, engine: Any | None) -> None:
        if engine is None:
            return

        setter = getattr(engine, "set_risk_context_provider", None)
        if callable(setter):
            try:
                setter(lambda: self)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug(
                    "Failed to configure execution risk context provider",
                    exc_info=True,
                )

    def _extract_gateway_reference(self) -> Mapping[str, object] | None:
        limits_payload = self._resolve_gateway_limits()
        if not isinstance(limits_payload, Mapping):
            return None
        candidate = limits_payload.get("risk_reference")
        if isinstance(candidate, Mapping):
            return candidate
        return None

    def _resolve_risk_metadata(self) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        try:
            summary = build_runtime_risk_metadata(self)
        except RiskApiError as exc:
            logger.debug("Trading risk metadata resolution failed", exc_info=True)
            return None, exc.to_metadata()
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            logger.debug("Unexpected risk metadata failure", exc_info=True)
            return (
                None,
                {
                    "message": "Trading risk metadata resolution failed",
                    "error": str(exc),
                    "runbook": RISK_API_RUNBOOK,
                },
            )
        return dict(summary), None

    @staticmethod
    def _merge_risk_reference(
        existing: Mapping[str, object] | None, addition: Mapping[str, object]
    ) -> dict[str, object]:
        return merge_risk_references(existing, addition)

    @staticmethod
    def _extract_symbol(intent: Any) -> str:
        for attr in ("symbol", "instrument", "asset"):
            value = getattr(intent, attr, None)
            if value:
                return str(value)
        if isinstance(intent, dict) and "symbol" in intent:
            return str(intent["symbol"])
        return "UNKNOWN"

    @staticmethod
    def _extract_strategy_id(intent: Any) -> str | None:
        for attr in ("strategy_id", "strategy"):
            value = getattr(intent, attr, None)
            if value:
                return str(value)
        if isinstance(intent, dict):
            for key in ("strategy_id", "strategy"):
                if key in intent and intent[key]:
                    return str(intent[key])
        return None

    @staticmethod
    def _extract_side(intent: Any) -> str | None:
        for attr in ("side", "direction", "order_side"):
            value = getattr(intent, attr, None)
            if value:
                return str(value).upper()
        if isinstance(intent, dict):
            for key in ("side", "direction", "order_side"):
                value = intent.get(key)
                if value:
                    return str(value).upper()
        metadata = getattr(intent, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("side")
            if value:
                return str(value).upper()
        return None

    @staticmethod
    def _extract_quantity(intent: Any) -> Decimal:
        for attr in ("quantity", "size", "volume"):
            value = getattr(intent, attr, None)
            if value is not None:
                return Decimal(str(value))
        if isinstance(intent, dict):
            for key in ("quantity", "size", "volume"):
                if key in intent:
                    return Decimal(str(intent[key]))
        return Decimal("0")

    @staticmethod
    def _extract_confidence(intent: Any) -> float | None:
        for attr in ("confidence", "score"):
            value = getattr(intent, attr, None)
            if value is not None:
                coerced = TradingManager._coerce_float(value)
                if coerced is not None:
                    return coerced
        if isinstance(intent, dict):
            for key in ("confidence", "score"):
                if key in intent:
                    coerced = TradingManager._coerce_float(intent[key])
                    if coerced is not None:
                        return coerced
        metadata = getattr(intent, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("confidence")
            coerced = TradingManager._coerce_float(value)
            if coerced is not None:
                return coerced
        return None

    @staticmethod
    def _extract_price(intent: Any, portfolio_state: Mapping[str, Any]) -> float:
        for attr in ("price", "limit_price", "entry_price"):
            value = getattr(intent, attr, None)
            if value is not None:
                coerced = coerce_float(value)
                if coerced is not None:
                    return coerced
        if isinstance(intent, dict):
            for key in ("price", "limit_price", "entry_price"):
                if key in intent:
                    coerced = coerce_float(intent[key])
                    if coerced is not None:
                        return coerced
        return coerce_float(portfolio_state.get("current_price", 0.0), default=0.0)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        return coerce_float(value)

    async def _emit_risk_interface_snapshot(self) -> None:
        """Resolve and publish the trading risk interface summary."""

        try:
            interface = resolve_trading_risk_interface(self)
        except RiskApiError as exc:
            alert = build_risk_interface_error(exc)
            self._last_risk_interface_snapshot = None
            self._last_risk_interface_error = alert
            logger.error(
                "Trading risk interface resolution failed: %s. See %s",
                alert.message,
                alert.runbook,
            )
            try:
                await publish_risk_interface_error(
                    self.event_bus,
                    alert,
                    source="trading_manager",
                )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to publish risk interface error telemetry",
                    exc_info=True,
                )
            return

        snapshot = build_risk_interface_snapshot(interface)
        self._last_risk_interface_snapshot = snapshot
        self._last_risk_interface_error = None

        logger.info("🛡️ Risk interface\n%s", format_risk_interface_markdown(snapshot))

        try:
            await publish_risk_interface_snapshot(
                self.event_bus,
                snapshot,
                source="trading_manager",
            )
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to publish risk interface telemetry",
                exc_info=True,
            )

    async def _emit_risk_snapshot(self) -> None:
        """Compute and publish the latest risk posture telemetry."""

        try:
            state = cast(Mapping[str, Any], self.portfolio_monitor.get_state())
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Unable to gather portfolio state for risk telemetry", exc_info=True)
            return

        limits_payload: Mapping[str, Any] | None = None
        last_decision: Mapping[str, Any] | None = None

        try:
            limits_payload = cast(Mapping[str, Any], self.risk_gateway.get_risk_limits())
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("RiskGateway.get_risk_limits failed", exc_info=True)

        try:
            last_decision = cast(Mapping[str, Any] | None, self.risk_gateway.get_last_decision())
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("RiskGateway.get_last_decision failed", exc_info=True)

        snapshot = evaluate_risk_posture(state, limits_payload, last_decision=last_decision)
        self._last_risk_snapshot = snapshot

        logger.info("🛡️ Risk posture snapshot\n%s", format_risk_markdown(snapshot))

        try:
            await publish_risk_snapshot(self.event_bus, snapshot, source="trading_manager")
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish risk telemetry", exc_info=True)

        try:
            roi_snapshot = evaluate_roi_posture(
                state,
                self._roi_cost_model,
                executed_trades=self._roi_executed_trades,
                total_notional=self._roi_total_notional,
                period_start=self._roi_period_start,
            )
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to compute ROI telemetry", exc_info=True)
        else:
            self._last_roi_snapshot = roi_snapshot
            logger.info("💹 ROI snapshot\n%s", format_roi_summary(roi_snapshot))
            try:
                await publish_roi_snapshot(self.event_bus, roi_snapshot, source="trading_manager")
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to publish ROI telemetry", exc_info=True)

    async def _emit_policy_snapshot(self) -> None:
        """Publish the most recent policy decision as telemetry."""

        snapshot: RiskPolicyEvaluationSnapshot | None = None
        try:
            snapshot = self.risk_gateway.get_last_policy_snapshot()
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("RiskGateway.get_last_policy_snapshot failed", exc_info=True)

        if snapshot is None and self._risk_policy is not None:
            try:
                policy_decision = self.risk_gateway.get_last_policy_decision()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("RiskGateway.get_last_policy_decision failed", exc_info=True)
                return
            if policy_decision is None:
                return
            snapshot = build_policy_snapshot(policy_decision, self._risk_policy)
        elif snapshot is None:
            return

        self._last_policy_snapshot = snapshot
        logger.info("🛡️ Policy decision\n%s", format_policy_markdown(snapshot))

        try:
            await publish_policy_snapshot(self.event_bus, snapshot, source="trading_manager")
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish risk policy telemetry", exc_info=True)

        if not snapshot.approved or snapshot.violations:
            severity = "critical" if not snapshot.approved else "warning"
            alert: RiskPolicyViolationAlert = build_policy_violation_alert(
                snapshot,
                severity=severity,
                runbook=RISK_POLICY_VIOLATION_RUNBOOK,
            )
            logger.warning("🚨 Policy violation alert\n%s", format_policy_violation_markdown(alert))
            try:
                await publish_policy_violation(
                    self.event_bus,
                    alert,
                    source="trading_manager",
                )
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to publish policy violation telemetry", exc_info=True)
