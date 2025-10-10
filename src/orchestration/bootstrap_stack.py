"""Bootstrap orchestration helpers that stitch the encyclopedia vision together."""

from __future__ import annotations

import inspect
import logging
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
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.trading_manager import TradingManager
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
        market_data = await self.fabric.fetch_latest(
            symbol, as_of=as_of, allow_stale=allow_stale, use_cache=False
        )
        synthesis = await self.fusion_engine.analyze_market_understanding(market_data)
        snapshot = SensorySnapshot(symbol=symbol, market_data=market_data, synthesis=synthesis)
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

    async def evaluate_tick(self, symbol: str, *, as_of: datetime | None = None) -> dict[str, Any]:
        snapshot = await self.pipeline.process_tick(symbol, as_of=as_of)
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

        market_price = float(getattr(snapshot.market_data, "close", snapshot.market_data.mid_price))
        intent = PaperTradeIntent(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            quantity=self.requested_quantity,
            price=market_price,
            confidence=float(snapshot.synthesis.confidence),
        )
        intent.metadata.setdefault("stop_loss_pct", self.stop_loss_pct)
        intent.metadata.setdefault(
            "intelligence_snapshot",
            {
                "unified_score": unified_score,
                "confidence": float(snapshot.synthesis.confidence),
                "narrative": snapshot.synthesis.dominant_narrative.value,
            },
        )

        await self.trading_manager.on_trade_intent(intent)
        decision = self.trading_manager.risk_gateway.get_last_decision()

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
            "status": "submitted",
            "liquidity_summary": liquidity_summary,
        }
        self.decisions.append(result_payload)
        self._notify_control_center(snapshot, result_payload)
        self._record_decision_diary(
            snapshot,
            intent,
            decision,
            liquidity_summary,
        )
        return result_payload

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
            metadata = {key: value for key, value in metadata.items() if value}

            self._diary_store.record(
                policy_id=policy_id,
                decision=policy_decision,
                regime_state=regime_state,
                outcomes=outcomes,
                metadata=metadata,
                notes=(
                    f"Bootstrap runtime decision for {snapshot.symbol}",
                ),
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
            describe_last_error = getattr(engine, "describe_last_error", None)
            if callable(describe_last_error):
                try:
                    error_snapshot = describe_last_error()
                except Exception:  # pragma: no cover - diagnostics only
                    error_snapshot = None
                if isinstance(error_snapshot, Mapping) and error_snapshot:
                    outcome["last_error"] = self._serialise(error_snapshot)

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
