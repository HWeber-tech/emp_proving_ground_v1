"""High-level runner that executes the full AlphaTrade loop pipeline.

The runner bridges the sensory → belief → regime → policy routing pipeline
with the governance instrumentation shipped in
``src.orchestration.alpha_trade_loop`` and the trading manager.  It
complements the roadmap deliverable that graduates the live-shadow pilot by
providing a cohesive service object that:

* Emits belief states from fused sensory snapshots
* Classifies the regime so fast-weight adapters receive calibrated confidence
* Executes the AlphaTrade loop orchestrator which records diary evidence and
  reflection artefacts
* Derives trade intents and forwards them to the trading manager while
  carrying the drift/thresh metadata produced by the orchestrator

The implementation is intentionally lightweight – it composes existing
components instead of inventing a new contract – so unit tests can exercise
end-to-end AlphaTrade runs without manual glue code inside the test fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    AlphaTradeLoopResult,
)
from src.understanding.belief import BeliefEmitter, BeliefState, RegimeFSM, RegimeSignal
from src.understanding.router import BeliefSnapshot, UnderstandingRouter


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True)
class TradePlan:
    """Trade metadata passed to the orchestrator and execution layer."""

    metadata: Mapping[str, Any] | None
    intent: Mapping[str, Any] | None


@dataclass(slots=True, frozen=True)
class AlphaTradeRunResult:
    """Composite artefacts emitted after executing a pipeline tick."""

    belief_state: BeliefState
    regime_signal: RegimeSignal
    loop_result: AlphaTradeLoopResult
    trade_metadata: Mapping[str, Any] | None
    trade_intent: Mapping[str, Any] | None


TradeBuilder = Callable[
    [Mapping[str, Any], BeliefState, RegimeSignal, Mapping[str, Any] | None],
    TradePlan,
]


class AlphaTradeLoopRunner:
    """Run a full AlphaTrade loop iteration and forward the trade intent."""

    def __init__(
        self,
        *,
        belief_emitter: BeliefEmitter,
        regime_fsm: RegimeFSM,
        orchestrator: AlphaTradeLoopOrchestrator,
        trading_manager: Any,
        understanding_router: UnderstandingRouter,
        publish_regime_signal: bool = False,
        trade_builder: TradeBuilder | None = None,
    ) -> None:
        self._belief_emitter = belief_emitter
        self._regime_fsm = regime_fsm
        self._orchestrator = orchestrator
        self._trading_manager = trading_manager
        self._understanding_router = understanding_router
        self._publish_regime_signal = publish_regime_signal
        self._trade_builder = trade_builder or self._default_trade_builder

    async def process(
        self,
        sensory_snapshot: Mapping[str, Any],
        *,
        regime_hint: str | None = None,
        policy_id: str | None = None,
        notes: Sequence[str] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        trade_overrides: Mapping[str, Any] | None = None,
    ) -> AlphaTradeRunResult:
        """Execute the AlphaTrade loop for a single fused sensory snapshot."""

        belief_state = self._belief_emitter.emit(sensory_snapshot, regime_hint=regime_hint)
        if self._publish_regime_signal:
            regime_signal = self._regime_fsm.publish(belief_state)
        else:
            regime_signal = self._regime_fsm.classify(belief_state)

        feature_flags = None
        flags_candidate = sensory_snapshot.get("feature_flags") if isinstance(sensory_snapshot, Mapping) else None
        if isinstance(flags_candidate, Mapping):
            feature_flags = {str(key): bool(value) for key, value in flags_candidate.items()}

        fast_weights_enabled = True
        fast_flag = sensory_snapshot.get("fast_weights_enabled") if isinstance(sensory_snapshot, Mapping) else None
        if isinstance(fast_flag, bool):
            fast_weights_enabled = fast_flag

        belief_snapshot = BeliefSnapshot(
            belief_id=belief_state.belief_id,
            regime_state=regime_signal.regime_state,
            features=dict(regime_signal.features),
            metadata={
                "signal_id": regime_signal.signal_id,
                "belief_generated_at": belief_state.generated_at.isoformat(),
                "symbol": belief_state.symbol,
            },
            fast_weights_enabled=fast_weights_enabled,
            feature_flags=feature_flags,
        )

        trade_plan = self._trade_builder(
            sensory_snapshot,
            belief_state,
            regime_signal,
            trade_overrides,
        )

        loop_result = self._orchestrator.run_iteration(
            belief_snapshot,
            belief_state=belief_state,
            policy_id=policy_id,
            trade=trade_plan.metadata,
            notes=tuple(notes or ()),
            extra_metadata=extra_metadata,
        )

        intent_payload = trade_plan.intent
        if intent_payload is None:
            intent_payload = self._build_trade_intent_from_decision(
                loop_result.decision,
                belief_state,
                regime_signal,
                trade_plan.metadata,
                trade_overrides,
            )

        if intent_payload is not None:
            await self._trading_manager.on_trade_intent(intent_payload)

        return AlphaTradeRunResult(
            belief_state=belief_state,
            regime_signal=regime_signal,
            loop_result=loop_result,
            trade_metadata=dict(trade_plan.metadata or {}),
            trade_intent=dict(intent_payload) if intent_payload is not None else None,
        )

    def _default_trade_builder(
        self,
        sensory_snapshot: Mapping[str, Any],
        belief_state: BeliefState,
        regime_signal: RegimeSignal,
        trade_overrides: Mapping[str, Any] | None,
    ) -> TradePlan:
        metadata: MutableMapping[str, Any] = {}
        if isinstance(trade_overrides, Mapping):
            metadata.update({str(key): value for key, value in trade_overrides.items()})

        tactic = None
        policy_id = metadata.get("policy_id") or trade_overrides.get("policy_id") if trade_overrides else None
        if policy_id:
            tactic = self._understanding_router.policy_router.tactics().get(str(policy_id))

        # Resolve baseline symbol/quantity/price from the tactic definition if present.
        if tactic is not None:
            parameters = tactic.parameters
            metadata.setdefault("symbol", parameters.get("symbol"))
            metadata.setdefault("side", parameters.get("side") or parameters.get("direction"))
            if "size" in parameters and "quantity" not in metadata:
                metadata.setdefault("quantity", parameters.get("size"))
            metadata.setdefault("quantity", parameters.get("quantity"))
            metadata.setdefault("price", parameters.get("price"))

        snapshot_symbol = sensory_snapshot.get("symbol") if isinstance(sensory_snapshot, Mapping) else None
        metadata.setdefault("symbol", snapshot_symbol or belief_state.symbol)

        if "side" not in metadata:
            action_hint = sensory_snapshot.get("action") if isinstance(sensory_snapshot, Mapping) else None
            metadata["side"] = action_hint or "hold"

        inferred_price = sensory_snapshot.get("price") if isinstance(sensory_snapshot, Mapping) else None
        if inferred_price is None:
            price_hint = sensory_snapshot.get("market_price") if isinstance(sensory_snapshot, Mapping) else None
            inferred_price = price_hint
        metadata.setdefault("price", inferred_price)

        confidence = regime_signal.regime_state.confidence
        metadata.setdefault("confidence", confidence)

        quantity_value = _coerce_float(metadata.get("quantity"))
        price_value = _coerce_float(metadata.get("price"))

        if quantity_value is None or quantity_value == 0.0:
            quantity_hint = sensory_snapshot.get("quantity") if isinstance(sensory_snapshot, Mapping) else None
            quantity_value = _coerce_float(quantity_hint)
            if quantity_value is not None:
                metadata["quantity"] = quantity_value

        if price_value is None and isinstance(sensory_snapshot, Mapping):
            bid = sensory_snapshot.get("bid")
            ask = sensory_snapshot.get("ask")
            price_value = _coerce_float(ask or bid)
            if price_value is not None:
                metadata["price"] = price_value

        if quantity_value is not None and price_value is not None:
            metadata.setdefault("notional", abs(quantity_value) * abs(price_value))

        timestamp_default = belief_state.generated_at
        if isinstance(timestamp_default, datetime):
            metadata.setdefault("timestamp", timestamp_default.isoformat())
        else:
            metadata.setdefault("timestamp", datetime.now().isoformat())
        metadata.setdefault("policy_id", policy_id or self._resolve_policy_id(metadata))

        # When we cannot derive a trade intent (no actionable side/quantity) we return metadata only.
        side_value = str(metadata.get("side") or "").strip().lower()
        if not side_value or side_value == "hold":
            return TradePlan(metadata=dict(metadata), intent=None)

        quantity_value = _coerce_float(metadata.get("quantity"))
        if not quantity_value:
            return TradePlan(metadata=dict(metadata), intent=None)

        price_value = _coerce_float(metadata.get("price"))

        intent_timestamp = belief_state.generated_at if isinstance(belief_state.generated_at, datetime) else datetime.now()

        intent: MutableMapping[str, Any] = {
            "strategy_id": metadata.get("policy_id"),
            "symbol": metadata.get("symbol"),
            "side": side_value.upper(),
            "quantity": quantity_value,
            "price": price_value,
            "confidence": metadata.get("confidence", confidence),
            "timestamp": intent_timestamp,
            "metadata": {
                "regime": regime_signal.regime_state.regime,
                "confidence": regime_signal.regime_state.confidence,
            },
        }

        release_hint = metadata.get("release_stage")
        if release_hint:
            intent["metadata"]["release_stage"] = release_hint

        notional_value = _coerce_float(metadata.get("notional"))
        if notional_value is not None:
            intent["metadata"]["notional"] = notional_value

        ticket_value = metadata.get("ticket")
        if ticket_value:
            intent["ticket"] = ticket_value

        return TradePlan(metadata=dict(metadata), intent=dict(intent))

    @staticmethod
    def _resolve_policy_id(metadata: Mapping[str, Any]) -> str | None:
        for key in ("policy_id", "strategy_id", "tactic_id"):
            value = metadata.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _build_trade_intent_from_decision(
        decision: Mapping[str, Any],
        belief_state: BeliefState,
        regime_signal: RegimeSignal,
        trade_metadata: Mapping[str, Any] | None,
        overrides: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        metadata: MutableMapping[str, Any] = {}
        if isinstance(trade_metadata, Mapping):
            metadata.update({str(key): value for key, value in trade_metadata.items()})
        if isinstance(overrides, Mapping):
            metadata.update({str(key): value for key, value in overrides.items()})

        metadata.setdefault("symbol", decision.get("parameters", {}).get("symbol") or belief_state.symbol)
        metadata.setdefault("policy_id", decision.get("tactic_id"))
        metadata.setdefault("side", decision.get("parameters", {}).get("side"))

        quantity_value = metadata.get("quantity")
        if quantity_value is None:
            parameters = decision.get("parameters", {})
            quantity_value = parameters.get("quantity") or parameters.get("size")
            if quantity_value is not None:
                metadata["quantity"] = quantity_value

        if metadata.get("price") is None:
            parameters = decision.get("parameters", {})
            price = parameters.get("price")
            if price is not None:
                metadata["price"] = price

        quantity = _coerce_float(metadata.get("quantity"))
        price = _coerce_float(metadata.get("price"))
        if quantity is not None and price is not None:
            metadata.setdefault("notional", abs(quantity) * abs(price))

        side = str(metadata.get("side") or "").strip().upper()
        if not side:
            return None
        if quantity is None or quantity == 0.0:
            return None

        timestamp_value = metadata.get("timestamp")
        if isinstance(timestamp_value, str):
            try:
                timestamp_parsed = datetime.fromisoformat(timestamp_value)
            except ValueError:
                timestamp_parsed = None
        elif isinstance(timestamp_value, datetime):
            timestamp_parsed = timestamp_value
        else:
            timestamp_parsed = None

        intent: MutableMapping[str, Any] = {
            "strategy_id": metadata.get("policy_id") or decision.get("tactic_id"),
            "symbol": metadata.get("symbol"),
            "side": side,
            "quantity": quantity,
            "price": price,
            "confidence": metadata.get("confidence")
            or metadata.get("regime_confidence")
            or regime_signal.regime_state.confidence,
            "timestamp": timestamp_parsed or belief_state.generated_at or datetime.now(),
            "metadata": {
                "regime": regime_signal.regime_state.regime,
                "confidence": regime_signal.regime_state.confidence,
            },
        }

        notional = _coerce_float(metadata.get("notional"))
        if notional is not None:
            intent["metadata"]["notional"] = notional

        ticket = metadata.get("ticket")
        if ticket:
            intent["ticket"] = ticket

        release_stage = metadata.get("release_stage")
        if release_stage:
            intent["metadata"]["release_stage"] = release_stage

        return intent


__all__ = [
    "AlphaTradeLoopRunner",
    "AlphaTradeRunResult",
    "TradePlan",
]
