"""Trading Manager v1.0 - Risk-Aware Trade Execution Coordinator."""

import logging
import time
from collections import deque
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, cast
from uuid import uuid4

try:  # pragma: no cover - redis optional in bootstrap deployments
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from src.config.risk.risk_config import RiskConfig as TradingRiskConfig
from src.core.coercion import coerce_float, coerce_int
from src.risk.real_risk_manager import RealRiskConfig, RealRiskManager
from src.risk.telemetry import (
    RiskTelemetrySnapshot,
    evaluate_risk_posture,
    format_risk_markdown,
    publish_risk_snapshot,
)
from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.policy_telemetry import (
    RiskPolicyEvaluationSnapshot,
    build_policy_snapshot,
    format_policy_markdown,
    publish_policy_snapshot,
)
from src.trading.risk.risk_gateway import RiskGateway
from src.trading.risk.risk_policy import RiskPolicy

from src.operations.roi import (
    RoiCostModel,
    RoiTelemetrySnapshot,
    evaluate_roi_posture,
    format_roi_markdown as format_roi_summary,
    publish_roi_snapshot,
)

try:
    from src.core.events import TradeIntent  # legacy
except Exception:  # pragma: no cover
    TradeIntent = TradeRejected = object
try:
    from src.core.risk.position_sizing import position_size as _PositionSizer  # canonical
except Exception:  # pragma: no cover
    _PositionSizer = None  # type: ignore[assignment]
# Provide precise callable typing for the sizer (Optional at runtime)
PositionSizer = cast(Optional[Callable[[Decimal, Decimal, Decimal], Decimal]], _PositionSizer)

logger = logging.getLogger(__name__)


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
        min_intent_confidence: float = 0.2,
        min_liquidity_confidence: float = 0.3,
        roi_cost_model: RoiCostModel | None = None,
        risk_config: TradingRiskConfig | None = None,
        risk_policy: RiskPolicy | None = None,
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
        """
        self.event_bus = event_bus
        self.strategy_registry = strategy_registry
        self.execution_engine = execution_engine

        # Initialize risk management components
        resolved_client = self._resolve_redis_client(redis_client)
        self.portfolio_monitor = PortfolioMonitor(event_bus, resolved_client)
        self.position_sizer = PositionSizer

        base_config = risk_config or TradingRiskConfig()
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
        risk_manager_config = RealRiskConfig(
            max_position_risk=float(effective_config.max_risk_per_trade_pct),
            max_total_exposure=float(effective_config.max_total_exposure_pct),
            max_drawdown=float(effective_config.max_drawdown_pct),
            max_leverage=float(effective_config.max_leverage),
            equity=float(initial_equity),
        )
        self._portfolio_risk_manager = RealRiskManager(risk_manager_config)
        self._risk_policy = risk_policy or RiskPolicy.from_config(effective_config)
        self._last_policy_snapshot: RiskPolicyEvaluationSnapshot | None = None

        risk_per_trade_decimal = Decimal(str(resolved_risk_per_trade))
        self.risk_gateway = RiskGateway(
            strategy_registry=strategy_registry,
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            risk_per_trade=risk_per_trade_decimal,
            max_open_positions=max_open_positions,
            max_daily_drawdown=resolved_drawdown,
            liquidity_prober=liquidity_prober,
            min_intent_confidence=min_intent_confidence,
            min_liquidity_confidence=min_liquidity_confidence,
            risk_policy=self._risk_policy,
            portfolio_risk_manager=self._portfolio_risk_manager,
        )

        self._last_risk_snapshot: RiskTelemetrySnapshot | None = None
        self._last_roi_snapshot: RoiTelemetrySnapshot | None = None

        base_model = roi_cost_model or RoiCostModel.bootstrap_defaults(initial_equity)
        self._roi_cost_model = base_model
        self._roi_period_start = datetime.now(tz=timezone.utc)
        self._roi_executed_trades = 0
        self._roi_total_notional = 0.0

        self._execution_stats: dict[str, object] = {
            "orders_submitted": 0,
            "orders_executed": 0,
            "orders_failed": 0,
            "latency_samples": 0,
            "total_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "last_error": None,
            "last_execution_at": None,
        }
        self._experiment_events: deque[dict[str, Any]] = deque(maxlen=512)

        logger.info(
            f"TradingManager initialized with equity={initial_equity}, "
            f"risk_per_trade={resolved_risk_per_trade * 100}%, "
            f"max_open_positions={max_open_positions}, "
            f"max_daily_drawdown={resolved_drawdown * 100}%"
        )
        logger.info("💹 ROI cost model configured: %s", self._roi_cost_model.as_dict())

    async def on_trade_intent(self, event: TradeIntent) -> None:
        """
        Handle incoming trade intents with integrated risk management.

        This method replaces the direct execution flow with a risk-validated
        pipeline. All trade intents must pass through the RiskGateway before
        reaching the execution engine.

        Args:
            event: The trade intent event to process
        """
        event_id = getattr(event, "event_id", getattr(event, "id", "unknown"))
        try:
            logger.info(f"Received trade intent: {event_id}")

            strategy_id = self._extract_strategy_id(event)
            base_symbol = self._extract_symbol(event)
            base_confidence = self._extract_confidence(event)

            portfolio_state = cast(Any, self.portfolio_monitor).get_state()

            validated_intent = await self.risk_gateway.validate_trade_intent(
                intent=event, portfolio_state=portfolio_state
            )

            if validated_intent:
                logger.info(
                    f"Trade intent {event_id} validated successfully. "
                    f"Calculated size: {self._extract_quantity(validated_intent)}"
                )

                symbol = self._extract_symbol(validated_intent) or base_symbol
                quantity = self._extract_quantity(validated_intent)
                price = self._extract_price(validated_intent, portfolio_state)
                cast(Any, self.portfolio_monitor).reserve_position(symbol, float(quantity), price)

                notional = abs(float(quantity)) * abs(float(price))

                start = time.perf_counter()
                current_submitted = coerce_int(
                    self._execution_stats.get("orders_submitted"), default=0
                )
                self._execution_stats["orders_submitted"] = current_submitted + 1
                try:
                    result = await self.execution_engine.process_order(validated_intent)
                except Exception as exc:
                    logger.exception("Execution engine error for trade intent %s", event_id)
                    self._record_execution_failure(exc)
                    decision = self._get_last_risk_decision()
                    failure_metadata: dict[str, object] = {"error": str(exc)}
                    if notional:
                        failure_metadata["notional"] = float(notional)
                    confidence = self._extract_confidence(validated_intent)
                    if confidence is None:
                        confidence = base_confidence
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
                else:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    self._record_execution_success(latency_ms, result)
                    decision = self._get_last_risk_decision()
                    confidence = self._extract_confidence(validated_intent)
                    if confidence is None:
                        confidence = base_confidence
                    if self._execution_stats.get("last_successful_order"):
                        if notional > 0:
                            self._roi_executed_trades += 1
                            self._roi_total_notional += notional
                        success_metadata: dict[str, object] = {
                            "latency_ms": float(latency_ms),
                            "notional": float(notional),
                            "quantity": float(quantity),
                            "price": float(price),
                        }
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
                    else:
                        error_reason = str(
                            self._execution_stats.get("last_error") or "execution_failed"
                        )
                        fallback_metadata: dict[str, object] = {"reason": error_reason}
                        if notional:
                            fallback_metadata["notional"] = float(notional)
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

            else:
                logger.warning(f"Trade intent {event_id} was rejected by the Risk Gateway")
                decision = self._get_last_risk_decision()
                reason = None
                if isinstance(decision, Mapping):
                    reason = decision.get("reason")
                rejection_metadata: dict[str, object] = {}
                if reason:
                    rejection_metadata["reason"] = reason
                self._record_experiment_event(
                    event_id=event_id,
                    status="rejected",
                    strategy_id=strategy_id,
                    symbol=base_symbol,
                    confidence=base_confidence,
                    metadata=rejection_metadata or None,
                    decision=decision,
                )

        except Exception as e:
            logger.error(f"Error processing trade intent {event_id}: {e}")
        finally:
            await self._emit_policy_snapshot()
            await self._emit_risk_snapshot()

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

    def _get_last_risk_decision(self) -> Mapping[str, Any] | None:
        """Safely fetch the last risk decision from the gateway."""

        try:
            return cast(Optional[Mapping[str, Any]], self.risk_gateway.get_last_decision())
        except Exception:
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
        return stats

    def get_experiment_events(self, limit: int | None = None) -> list[Mapping[str, Any]]:
        """Expose the recent paper-trading experiment events."""

        events: list[Mapping[str, Any]] = [dict(event) for event in self._experiment_events]
        if limit is None:
            return events
        count = int(limit)
        if count <= 0:
            return []
        return events[: min(len(events), count)]

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
        payload: dict[str, object] = {
            "risk_limits": cast(Any, self.risk_gateway).get_risk_limits(),
            "portfolio_state": cast(Any, self.portfolio_monitor).get_state(),
        }
        last_decision = cast(Any, self.risk_gateway).get_last_decision()
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
        if hasattr(self, "_risk_config"):
            try:
                payload["risk_config"] = self._risk_config.dict()
            except Exception:
                payload["risk_config"] = {}
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

        try:
            policy_decision = self.risk_gateway.get_last_policy_decision()
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("RiskGateway.get_last_policy_decision failed", exc_info=True)
            return

        if policy_decision is None or self._risk_policy is None:
            return

        snapshot = build_policy_snapshot(policy_decision, self._risk_policy)
        self._last_policy_snapshot = snapshot
        logger.info("🛡️ Policy decision\n%s", format_policy_markdown(snapshot))

        try:
            await publish_policy_snapshot(self.event_bus, snapshot, source="trading_manager")
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish risk policy telemetry", exc_info=True)
