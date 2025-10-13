"""Bootstrap orchestration helpers that stitch the encyclopedia vision together."""

from __future__ import annotations

import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from itertools import islice
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from src.core.base import MarketData
from src.data_foundation.fabric.market_data_fabric import MarketDataFabric
from src.orchestration.enhanced_understanding_engine import (
    ContextualFusionEngine,
    Synthesis,
)
from src.orchestration.decision_latency_guard import evaluate_decision_latency
from src.orchestration.pipeline_metrics import PipelineLatencyMonitor
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.trading_manager import TradingManager, TradeIntentOutcome
from src.thinking.adaptation.policy_router import PolicyDecision
from src.understanding.decision_diary import DecisionDiaryStore

if TYPE_CHECKING:
    from src.operations.bootstrap_control_center import BootstrapControlCenter


logger = logging.getLogger(__name__)


@dataclass
class SensorySnapshot:
    """Container representing a fused sensory observation."""

    symbol: str
    market_data: MarketData
    synthesis: Synthesis
    generated_at: datetime = field(default_factory=datetime.utcnow)
    latencies: Mapping[str, float] | None = None


class BootstrapSensoryPipeline:
    """Stream symbols through the market data fabric and fusion engine."""

    def __init__(
        self,
        fabric: MarketDataFabric,
        fusion_engine: ContextualFusionEngine | None = None,
    ) -> None:
        self.fabric = fabric
        self.fusion_engine = fusion_engine or ContextualFusionEngine()
        self.history: Dict[str, List[SensorySnapshot]] = {}
        self._listeners: list[Callable[[SensorySnapshot], Awaitable[None] | None]] = []
        self._audit_trail: deque[dict[str, Any]] = deque(maxlen=128)

    def register_listener(
        self, listener: Callable[[SensorySnapshot], Awaitable[None] | None]
    ) -> None:
        self._listeners.append(listener)

    async def process_tick(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
        allow_stale: bool = True,
    ) -> SensorySnapshot:
        ingest_start = time.perf_counter()
        market_data = await self.fabric.fetch_latest(
            symbol, as_of=as_of, allow_stale=allow_stale, use_cache=False
        )
        ingest_latency = time.perf_counter() - ingest_start
        signal_start = time.perf_counter()
        synthesis = await self.fusion_engine.analyze_market_understanding(market_data)
        signal_latency = time.perf_counter() - signal_start
        snapshot = SensorySnapshot(
            symbol=symbol,
            market_data=market_data,
            synthesis=synthesis,
            latencies={
                "ingest": ingest_latency,
                "signal": signal_latency,
            },
        )
        timestamp = getattr(market_data, "timestamp", None)
        if isinstance(timestamp, datetime):
            snapshot.generated_at = (
                timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
            )
        self.history.setdefault(symbol, []).append(snapshot)

        self._record_audit_entry(symbol, synthesis)

        for listener in list(self._listeners):
            try:
                outcome = listener(snapshot)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "Bootstrap sensory listener failed",  # noqa: TRY401 - intentional logging path
                    extra={
                        "listener": getattr(listener, "__name__", repr(listener)),
                        "symbol": symbol,
                    },
                )
                continue

        return snapshot

    def _record_audit_entry(self, symbol: str, synthesis: Synthesis) -> None:
        readings = getattr(self.fusion_engine, "current_readings", {})
        if not readings:
            return

        dimensions: dict[str, dict[str, Any]] = {}
        for name, reading in readings.items():
            dimensions[name] = {
                "signal": float(getattr(reading, "signal_strength", 0.0)),
                "confidence": float(getattr(reading, "confidence", 0.0)),
                "regime": getattr(getattr(reading, "regime", None), "name", None),
                "data_quality": float(getattr(reading, "data_quality", 0.0)),
            }

        entry = {
            "symbol": symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": float(synthesis.unified_score),
            "confidence": float(synthesis.confidence),
            "dimensions": dimensions,
        }
        self._audit_trail.appendleft(entry)

    def audit_trail(self, limit: int = 5) -> list[Mapping[str, Any]]:
        return [dict(item) for item in islice(self._audit_trail, 0, max(0, limit))]


@dataclass
class PaperTradeIntent:
    """Simple trade intent compatible with the risk gateway and trading manager."""

    strategy_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: float
    confidence: float
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BootstrapTradingStack:
    """Tie the sensory pipeline to the risk-aware trading manager."""

    def __init__(
        self,
        pipeline: BootstrapSensoryPipeline,
        trading_manager: TradingManager,
        *,
        strategy_id: str = "bootstrap-strategy",
        buy_threshold: float = 0.35,
        sell_threshold: float = 0.35,
        requested_quantity: Decimal | float = Decimal("1"),
        stop_loss_pct: float = 0.01,
        liquidity_prober: Any | None = None,
        control_center: "BootstrapControlCenter" | None = None,
        diary_store: DecisionDiaryStore | None = None,
        release_router: ReleaseAwareExecutionRouter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.trading_manager = trading_manager
        self.strategy_id = strategy_id
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)
        self.requested_quantity = (
            requested_quantity
            if isinstance(requested_quantity, Decimal)
            else Decimal(str(requested_quantity))
        )
        self.stop_loss_pct = float(stop_loss_pct)
        self.decisions: list[dict[str, Any]] = []
        self.liquidity_prober = liquidity_prober
        self.control_center = control_center
        self._diary_store = diary_store
        self._release_router = release_router
        self._latency_monitor = PipelineLatencyMonitor()

        if liquidity_prober is not None:

            async def _record(snapshot: SensorySnapshot) -> None:
                try:
                    liquidity_prober.record_snapshot(snapshot.symbol, snapshot.market_data)
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception(
                        "Liquidity prober snapshot recording failed",
                        extra={
                            "symbol": snapshot.symbol,
                            "prober": liquidity_prober.__class__.__name__,
                        },
                    )

            self.pipeline.register_listener(_record)

    def describe_pipeline_observability(self) -> Mapping[str, Any]:
        """Return heartbeat and latency percentiles for the bootstrap pipeline."""

        snapshot = self._latency_monitor.snapshot()
        latency_payload = {stage: dict(metrics) for stage, metrics in snapshot.latency.items()}
        return {
            "heartbeat": dict(snapshot.heartbeat),
            "latency": latency_payload,
            "decision_latency": evaluate_decision_latency(latency_payload.get("total")),
        }

    @property
    def pipeline_latency_monitor(self) -> PipelineLatencyMonitor:
        """Expose the underlying latency monitor for advanced integrations."""

        return self._latency_monitor

    async def evaluate_tick(self, symbol: str, *, as_of: datetime | None = None) -> dict[str, Any]:
        tick_timestamp = datetime.now(timezone.utc)
        latency_metrics: dict[str, float] = {}
        order_attempted = False

        try:
            snapshot = await self.pipeline.process_tick(symbol, as_of=as_of)
            if isinstance(snapshot.latencies, Mapping):
                for stage, value in snapshot.latencies.items():
                    if value is None:
                        continue
                    try:
                        latency_metrics[str(stage)] = float(value)
                    except (TypeError, ValueError):
                        continue

            unified_score = float(snapshot.synthesis.unified_score)

            side: Optional[str] = None
            if unified_score >= self.buy_threshold:
                side = "BUY"
            elif unified_score <= -self.sell_threshold:
                side = "SELL"

            if side is None:
                skip_result: dict[str, object | None] = {
                    "snapshot": snapshot,
                    "intent": None,
                    "decision": None,
                    "status": "skipped",
                }
                self.decisions.append(skip_result)
                self._notify_control_center(snapshot, skip_result)
                return skip_result

            order_prepare_start = time.perf_counter()
            market_price = float(
                getattr(snapshot.market_data, "close", snapshot.market_data.mid_price)
            )
            intent = PaperTradeIntent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                quantity=self.requested_quantity,
                price=market_price,
                confidence=float(snapshot.synthesis.confidence),
            )
            intent.metadata.setdefault("stop_loss_pct", self.stop_loss_pct)
            understanding_snapshot = {
                "unified_score": unified_score,
                "confidence": float(snapshot.synthesis.confidence),
                "narrative": snapshot.synthesis.dominant_narrative.value,
            }
            intent.metadata.setdefault("understanding_snapshot", understanding_snapshot)
            if isinstance(intent.metadata, MutableMapping):
                intent.metadata.pop("intelligence_snapshot", None)
            latency_metrics["order"] = time.perf_counter() - order_prepare_start

            ack_start = time.perf_counter()
            order_attempted = True
            try:
                trade_outcome = await self.trading_manager.on_trade_intent(intent)
            finally:
                latency_metrics["ack"] = time.perf_counter() - ack_start

            decision = self.trading_manager.risk_gateway.get_last_decision()

            outcome_payload: Mapping[str, Any] | None = None
            status = "submitted"
            if isinstance(trade_outcome, TradeIntentOutcome):
                outcome_payload = trade_outcome.as_dict()
                outcome_status = outcome_payload.get("status")
                if isinstance(outcome_status, str) and outcome_status.strip():
                    status = outcome_status.strip()
                else:
                    status = trade_outcome.status

            liquidity_summary = None
            if isinstance(decision, Mapping):
                for check in decision.get("checks", []):
                    if isinstance(check, Mapping) and check.get("name") == "liquidity_probe":
                        liquidity_summary = check.get("summary")
                        break

            result_payload: dict[str, object | None] = {
                "snapshot": snapshot,
                "intent": intent,
                "decision": decision,
                "status": status,
                "liquidity_summary": liquidity_summary,
            }
            if outcome_payload is not None:
                result_payload["trade_outcome"] = outcome_payload
                throttle_fragment = outcome_payload.get("throttle")
                if isinstance(throttle_fragment, Mapping):
                    result_payload["throttle"] = throttle_fragment
                outcome_metadata = outcome_payload.get("metadata")
                if isinstance(outcome_metadata, Mapping):
                    result_payload["trade_metadata"] = outcome_metadata

            self.decisions.append(result_payload)
            self._notify_control_center(snapshot, result_payload)
            self._record_decision_diary(
                snapshot,
                intent,
                decision,
                liquidity_summary,
                trade_outcome,
            )
            return result_payload
        finally:
            total_latency = sum(
                value
                for value in latency_metrics.values()
                if isinstance(value, (int, float)) and value >= 0.0
            )
            if total_latency > 0:
                latency_metrics.setdefault("total", total_latency)
            self._latency_monitor.observe_tick(
                latency_metrics,
                timestamp=tick_timestamp,
                order_attempted=order_attempted,
            )

    def set_release_router(
        self, router: ReleaseAwareExecutionRouter | None
    ) -> None:
        """Attach or replace the release-aware execution router reference."""

        self._release_router = router

    # ------------------------------------------------------------------
    # Decision diary integration
    # ------------------------------------------------------------------
    def _record_decision_diary(
        self,
        snapshot: SensorySnapshot,
        intent: PaperTradeIntent,
        decision: Mapping[str, Any] | None,
        liquidity_summary: Mapping[str, Any] | None,
        trade_outcome: TradeIntentOutcome | None,
    ) -> None:
        if self._diary_store is None:
            return

        try:
            policy_id = intent.strategy_id
            release_metadata = self._build_release_metadata(intent)
            execution_outcome = self._build_execution_outcome()
            risk_outcome = (
                {str(key): value for key, value in decision.items()}
                if isinstance(decision, Mapping)
                else {}
            )

            notes: list[str] = [f"Bootstrap runtime decision for {snapshot.symbol}"]

            trade_outcome_payload: Mapping[str, Any] | None = None
            trade_outcome_metadata: Mapping[str, Any] | None = None
            throttle_snapshot: Mapping[str, Any] | None = None
            if isinstance(trade_outcome, TradeIntentOutcome):
                trade_outcome_payload = trade_outcome.as_dict()
                metadata_candidate = trade_outcome_payload.get("metadata")
                if isinstance(metadata_candidate, Mapping):
                    trade_outcome_metadata = metadata_candidate
                throttle_candidate = trade_outcome.throttle
                if throttle_candidate is None:
                    throttle_candidate = trade_outcome_payload.get("throttle")
                if isinstance(throttle_candidate, Mapping):
                    throttle_snapshot = dict(throttle_candidate)

                outcome_status = trade_outcome_payload.get("status")
                if isinstance(outcome_status, str):
                    status_note = outcome_status.strip()
                    if status_note and status_note not in {"submitted", "executed"}:
                        notes.append(f"Trade outcome status: {status_note}")

            guardrails: dict[str, Any] = {
                "stop_loss_pct": float(intent.metadata.get("stop_loss_pct", self.stop_loss_pct))
                if isinstance(intent.metadata, Mapping)
                else self.stop_loss_pct
            }
            if isinstance(liquidity_summary, Mapping) and liquidity_summary:
                guardrails["liquidity_summary"] = self._serialise(liquidity_summary)

            reflection_summary: dict[str, Any] = {
                "narrative": snapshot.synthesis.dominant_narrative.value,
            }
            if isinstance(liquidity_summary, Mapping):
                reflection_summary["liquidity"] = self._serialise(liquidity_summary)

            policy_decision = PolicyDecision(
                tactic_id=policy_id,
                parameters={
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "quantity": float(intent.quantity),
                    "price": float(intent.price),
                    "confidence": float(intent.confidence),
                },
                selected_weight=float(snapshot.synthesis.unified_score),
                guardrails=guardrails,
                rationale="Bootstrap runtime threshold trigger",
                experiments_applied=(),
                reflection_summary=reflection_summary,
                decision_timestamp=snapshot.generated_at,
            )

            regime_state = {
                "regime": release_metadata.get("stage") or "bootstrap",
                "confidence": float(snapshot.synthesis.confidence),
                "features": {
                    "unified_score": float(snapshot.synthesis.unified_score),
                    "dominant_narrative": snapshot.synthesis.dominant_narrative.value,
                },
                "timestamp": snapshot.generated_at.isoformat(),
            }

            outcomes = {
                "risk": self._serialise(risk_outcome) if risk_outcome else None,
                "release": self._serialise(release_metadata)
                if release_metadata
                else None,
                "execution": self._serialise(execution_outcome)
                if execution_outcome
                else None,
            }
            if trade_outcome_payload:
                outcomes["trade_outcome"] = self._serialise(trade_outcome_payload)

            serialised_throttle: Mapping[str, Any] | None = None
            if throttle_snapshot is None:
                fallback_throttle = self.trading_manager.get_trade_throttle_snapshot()
                if isinstance(fallback_throttle, Mapping):
                    throttle_snapshot = dict(fallback_throttle)

            throttle_note: str | None = None
            if isinstance(throttle_snapshot, Mapping) and throttle_snapshot:
                message_value = throttle_snapshot.get("message")
                reason_value = throttle_snapshot.get("reason")
                state_value = throttle_snapshot.get("state")
                active_flag = bool(throttle_snapshot.get("active"))
                state_guard = str(state_value or "").lower()
                if active_flag or state_guard in {"rate_limited", "cooldown", "min_interval"}:
                    if isinstance(message_value, str) and message_value.strip():
                        throttle_note = message_value.strip()
                    elif isinstance(reason_value, str) and reason_value.strip():
                        throttle_note = f"Trade throttle active ({reason_value.strip()})"
                    elif isinstance(state_value, str) and state_value.strip():
                        throttle_note = f"Trade throttle active ({state_value.strip()})"
                    else:
                        throttle_note = "Trade throttle active"

                serialised_throttle = self._serialise(throttle_snapshot)
                outcomes["throttle"] = serialised_throttle

            outcomes = {key: value for key, value in outcomes.items() if value}

            metadata = {
                "snapshot": {
                    "symbol": snapshot.symbol,
                    "generated_at": snapshot.generated_at.isoformat(),
                },
                "intent_metadata": self._serialise(intent.metadata)
                if isinstance(intent.metadata, Mapping)
                else None,
            }
            if trade_outcome_metadata:
                metadata["trade_outcome_metadata"] = self._serialise(trade_outcome_metadata)
            if serialised_throttle is not None:
                metadata["trade_throttle"] = serialised_throttle

            metadata = {key: value for key, value in metadata.items() if value}

            if throttle_note:
                notes.append(throttle_note)

            self._diary_store.record(
                policy_id=policy_id,
                decision=policy_decision,
                regime_state=regime_state,
                outcomes=outcomes,
                metadata=metadata,
                notes=tuple(notes),
            )
        except Exception:  # pragma: no cover - diary logging must never break runtime
            logger.debug(
                "Bootstrap decision diary recording failed",
                exc_info=True,
            )

    def _build_release_metadata(
        self, intent: PaperTradeIntent
    ) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {}
        intent_meta = intent.metadata if isinstance(intent.metadata, Mapping) else None
        if intent_meta:
            metadata.update({str(key): value for key, value in intent_meta.items()})

        router = self._resolve_release_router()
        if router is not None:
            try:
                last_route = router.last_route()
            except Exception:  # pragma: no cover - diagnostics only
                last_route = None
            if isinstance(last_route, Mapping):
                metadata.setdefault("stage", last_route.get("stage"))
                metadata.setdefault("route", last_route.get("route"))
                metadata.setdefault("last_route", dict(last_route))
        if "stage" not in metadata:
            try:
                posture = self.trading_manager.describe_release_posture(
                    intent.strategy_id
                )
            except Exception:  # pragma: no cover - diagnostics only
                posture = None
            if isinstance(posture, Mapping):
                metadata.setdefault("stage", posture.get("stage"))
                metadata.setdefault("thresholds", posture.get("thresholds"))
        return metadata

    def _build_execution_outcome(self) -> Mapping[str, Any]:
        stats = self.trading_manager.get_execution_stats()
        outcome: dict[str, Any] = {}
        if stats:
            outcome["stats"] = self._serialise(stats)

        router = self._resolve_release_router()
        engine: Any | None = None
        if router is not None:
            engine = router.live_engine or router.paper_engine
        else:
            engine = self.trading_manager.execution_engine

        describe_last_order = getattr(engine, "describe_last_order", None)
        if callable(describe_last_order):
            try:
                snapshot = describe_last_order()
            except Exception:  # pragma: no cover - diagnostics only
                snapshot = None
            if isinstance(snapshot, Mapping):
                outcome["last_order"] = self._serialise(snapshot)

        if engine is not None:
            describe_metrics = getattr(engine, "describe_metrics", None)
            if callable(describe_metrics):
                try:
                    metrics_snapshot = describe_metrics()
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to capture paper execution metrics for diary entry",
                        exc_info=True,
                    )
                else:
                    if isinstance(metrics_snapshot, Mapping):
                        outcome["paper_metrics"] = self._serialise(metrics_snapshot)

            describe_last_error = getattr(engine, "describe_last_error", None)
            if callable(describe_last_error):
                try:
                    error_snapshot = describe_last_error()
                except Exception:  # pragma: no cover - diagnostics only
                    error_snapshot = None
                if isinstance(error_snapshot, Mapping) and error_snapshot:
                    outcome["last_error"] = self._serialise(error_snapshot)

            describe_broker = getattr(engine, "describe_broker", None)
            if callable(describe_broker):
                try:
                    broker_snapshot = describe_broker()
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to capture paper broker summary for diary entry",
                        exc_info=True,
                    )
                else:
                    if isinstance(broker_snapshot, Mapping) and broker_snapshot:
                        outcome["paper_broker"] = self._serialise(broker_snapshot)

        return outcome

    def _resolve_release_router(self) -> ReleaseAwareExecutionRouter | None:
        if self._release_router is not None:
            return self._release_router
        engine = self.trading_manager.execution_engine
        if isinstance(engine, ReleaseAwareExecutionRouter):
            return engine
        return None

    def _serialise(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, Mapping):
            return {str(key): self._serialise(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._serialise(item) for item in value]
        return value

    def _notify_control_center(self, snapshot: SensorySnapshot, result: Mapping[str, Any]) -> None:
        if self.control_center is not None:
            try:
                self.control_center.record_tick(snapshot=snapshot, result=result)
            except Exception:  # pragma: no cover - defensive guard
                logger.exception(
                    "Bootstrap control center notification failed",
                    extra={
                        "control_center": self.control_center.__class__.__name__,
                        "symbol": snapshot.symbol,
                    },
                )


__all__ = [
    "BootstrapSensoryPipeline",
    "BootstrapTradingStack",
    "PaperTradeIntent",
    "SensorySnapshot",
]
