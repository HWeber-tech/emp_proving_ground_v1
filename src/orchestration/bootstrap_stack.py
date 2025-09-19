"""Bootstrap orchestration helpers that stitch the encyclopedia vision together."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, TYPE_CHECKING

from src.core.base import MarketData
from src.data_foundation.fabric.market_data_fabric import MarketDataFabric
from src.orchestration.enhanced_intelligence_engine import ContextualFusionEngine, Synthesis
from src.trading.trading_manager import TradingManager

if TYPE_CHECKING:
    from src.operations.bootstrap_control_center import BootstrapControlCenter


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
        synthesis = await self.fusion_engine.analyze_market_intelligence(market_data)
        snapshot = SensorySnapshot(symbol=symbol, market_data=market_data, synthesis=synthesis)
        self.history.setdefault(symbol, []).append(snapshot)

        for listener in list(self._listeners):
            try:
                outcome = listener(snapshot)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:
                # Observability callbacks should never break the pipeline
                continue

        return snapshot


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

        if liquidity_prober is not None:
            async def _record(snapshot: SensorySnapshot) -> None:
                try:
                    liquidity_prober.record_snapshot(snapshot.symbol, snapshot.market_data)
                except Exception:
                    # Observability helpers must never break the decision loop
                    pass

            self.pipeline.register_listener(_record)

    async def evaluate_tick(
        self, symbol: str, *, as_of: datetime | None = None
    ) -> dict[str, Any]:
        snapshot = await self.pipeline.process_tick(symbol, as_of=as_of)
        unified_score = float(snapshot.synthesis.unified_score)

        side: Optional[str] = None
        if unified_score >= self.buy_threshold:
            side = "BUY"
        elif unified_score <= -self.sell_threshold:
            side = "SELL"

        if side is None:
            result = {"snapshot": snapshot, "intent": None, "decision": None, "status": "skipped"}
            self.decisions.append(result)
            self._notify_control_center(snapshot, result)
            return result

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

        result = {
            "snapshot": snapshot,
            "intent": intent,
            "decision": decision,
            "status": "submitted",
            "liquidity_summary": liquidity_summary,
        }
        self.decisions.append(result)
        self._notify_control_center(snapshot, result)
        return result

    def _notify_control_center(self, snapshot: SensorySnapshot, result: Mapping[str, Any]) -> None:
        if self.control_center is not None:
            try:
                self.control_center.record_tick(snapshot=snapshot, result=result)
            except Exception:
                # Telemetry hooks must never impact the trading decision flow
                pass


__all__ = [
    "BootstrapSensoryPipeline",
    "BootstrapTradingStack",
    "PaperTradeIntent",
    "SensorySnapshot",
]
