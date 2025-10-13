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

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import inspect
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Sequence

from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    AlphaTradeLoopResult,
)
from src.understanding.belief import BeliefEmitter, BeliefState, RegimeFSM, RegimeSignal
from src.understanding.router import BeliefSnapshot, UnderstandingRouter

if TYPE_CHECKING:
    from src.trading.trading_manager import TradeIntentOutcome
    from src.understanding.decision_diary import DecisionDiaryEntry


logger = logging.getLogger(__name__)


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
    trade_outcome: "TradeIntentOutcome | None"


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

        fast_weight_metadata = self._build_fast_weight_metadata(
            loop_result.decision_bundle,
            belief_snapshot,
        )

        trade_metadata = dict(trade_plan.metadata or {})
        if "fast_weight" not in trade_metadata:
            trade_metadata["fast_weight"] = dict(fast_weight_metadata)
        guardrails_payload = dict(loop_result.decision.guardrails)
        if guardrails_payload:
            trade_metadata.setdefault("guardrails", guardrails_payload)

        attribution_payload = self._build_order_attribution(
            belief_state=belief_state,
            decision_bundle=loop_result.decision_bundle,
            diary_entry=loop_result.diary_entry,
        )
        if attribution_payload:
            trade_metadata.setdefault("attribution", attribution_payload)

        intent_payload = None
        if trade_plan.intent is not None:
            raw_intent = dict(trade_plan.intent)
            metadata_payload = dict(raw_intent.get("metadata", {}))
            if "fast_weight" not in metadata_payload:
                metadata_payload["fast_weight"] = dict(fast_weight_metadata)
            if attribution_payload and "attribution" not in metadata_payload:
                metadata_payload["attribution"] = attribution_payload
            if guardrails_payload and "guardrails" not in metadata_payload:
                metadata_payload["guardrails"] = guardrails_payload
            raw_intent["metadata"] = metadata_payload
            intent_payload = raw_intent
        if intent_payload is None:
            intent_payload = self._build_trade_intent_from_decision(
                loop_result.decision,
                belief_state,
                regime_signal,
                trade_metadata,
                trade_overrides,
            )

        trade_outcome: "TradeIntentOutcome | None" = None
        diary_annotations: dict[str, Any] = {}
        loop_metadata_updates: dict[str, Any] = {}
        if attribution_payload:
            diary_annotations["attribution"] = attribution_payload
            loop_metadata_updates["attribution"] = attribution_payload
        if intent_payload is not None:
            outcome = await self._trading_manager.on_trade_intent(intent_payload)
            trade_outcome = outcome
            if outcome is not None:
                trade_execution_payload: dict[str, Any] = {
                    "status": outcome.status,
                    "executed": outcome.executed,
                }
                if outcome.metadata:
                    trade_execution_payload["metadata"] = dict(outcome.metadata)
                if outcome.throttle:
                    trade_execution_payload["throttle"] = dict(outcome.throttle)
                diary_annotations["trade_execution"] = trade_execution_payload
                loop_metadata_updates["trade_execution"] = trade_execution_payload

        performance_health = await self._collect_performance_health()
        if performance_health is not None:
            diary_annotations.setdefault("performance_health", performance_health)
            loop_metadata_updates.setdefault("performance_health", performance_health)
            trade_metadata.setdefault("performance_health", performance_health)

        if diary_annotations:
            merged_loop_metadata = dict(loop_result.metadata)
            merged_loop_metadata.update(loop_metadata_updates)
            updated_entry = self._orchestrator.annotate_diary_entry(
                loop_result.diary_entry.entry_id,
                diary_annotations,
            )
            loop_result = replace(
                loop_result,
                diary_entry=updated_entry,
                metadata=MappingProxyType(merged_loop_metadata),
            )

        if loop_result.metadata.get("trade_metadata") != trade_metadata:
            merged_loop_metadata = dict(loop_result.metadata)
            merged_loop_metadata["trade_metadata"] = dict(trade_metadata)
            loop_result = replace(
                loop_result,
                metadata=MappingProxyType(merged_loop_metadata),
            )

        return AlphaTradeRunResult(
            belief_state=belief_state,
            regime_signal=regime_signal,
            loop_result=loop_result,
            trade_metadata=dict(trade_metadata),
            trade_intent=dict(intent_payload) if intent_payload is not None else None,
            trade_outcome=trade_outcome,
        )

    @staticmethod
    def _select_top_features(
        features: Mapping[str, Any] | None,
        *,
        limit: int = 4,
    ) -> list[Mapping[str, Any]]:
        if not isinstance(features, Mapping):
            return []
        ranked: list[tuple[str, float]] = []
        for name, value in features.items():
            try:
                ranked.append((str(name), float(value)))
            except (TypeError, ValueError):
                continue
        ranked.sort(key=lambda item: abs(item[1]), reverse=True)
        summary: list[Mapping[str, Any]] = []
        for name, value in ranked[:limit]:
            summary.append({"name": name, "value": value})
        return summary

    def _build_order_attribution(
        self,
        *,
        belief_state: BeliefState,
        decision_bundle: UnderstandingDecision,
        diary_entry: DecisionDiaryEntry,
    ) -> Mapping[str, Any] | None:
        regime_state = decision_bundle.belief_snapshot.regime_state
        belief_summary: dict[str, Any] = {
            "belief_id": belief_state.belief_id,
            "symbol": belief_state.symbol,
            "regime": regime_state.regime,
            "confidence": float(regime_state.confidence),
        }

        generated_at = belief_state.generated_at
        if isinstance(generated_at, datetime):
            belief_summary["generated_at"] = generated_at.astimezone(timezone.utc).isoformat()

        metadata = getattr(belief_state, "metadata", None)
        if isinstance(metadata, Mapping) and metadata:
            belief_summary["metadata"] = {str(key): value for key, value in metadata.items()}

        top_features = self._select_top_features(decision_bundle.belief_snapshot.features)
        if top_features:
            belief_summary["top_features"] = top_features

        probes_payload: list[Mapping[str, Any]] = []
        for activation in diary_entry.probes:
            probe_entry: dict[str, Any] = {
                "probe_id": activation.probe_id,
                "status": activation.status,
            }
            if activation.severity:
                probe_entry["severity"] = activation.severity
            if activation.owner:
                probe_entry["owner"] = activation.owner
            if activation.contact:
                probe_entry["contact"] = activation.contact
            if activation.runbook:
                probe_entry["runbook"] = activation.runbook
            if activation.notes:
                probe_entry["notes"] = list(activation.notes)
            if activation.metadata:
                probe_entry["metadata"] = dict(activation.metadata)
            probes_payload.append(probe_entry)

        explanation = (decision_bundle.decision.rationale or "").strip()
        if not explanation:
            explanation = (
                f"{decision_bundle.decision.tactic_id} routed under {regime_state.regime}"
            )

        attribution: dict[str, Any] = {
            "diary_entry_id": diary_entry.entry_id,
            "policy_id": diary_entry.policy_id,
            "belief": belief_summary,
            "probes": probes_payload,
            "explanation": explanation,
        }
        return attribution

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
    def _build_fast_weight_metadata(
        decision_bundle: UnderstandingDecision,
        belief_snapshot: BeliefSnapshot,
    ) -> Mapping[str, Any]:
        summary_payload = {
            adapter_id: dict(summary)
            for adapter_id, summary in decision_bundle.fast_weight_summary.items()
        }
        metrics_payload = dict(decision_bundle.fast_weight_metrics)
        enabled_flag = belief_snapshot.fast_weights_enabled
        enabled_value = enabled_flag if isinstance(enabled_flag, bool) else None
        return {
            "enabled": enabled_value,
            "metrics": metrics_payload,
            "summary": summary_payload,
            "applied_adapters": list(decision_bundle.applied_adapters),
        }

    async def _collect_performance_health(self) -> Mapping[str, Any] | None:
        """Fetch and normalise the trading manager's performance health snapshot."""

        assessor = getattr(self._trading_manager, "assess_performance_health", None)
        if assessor is None or not callable(assessor):
            return None

        try:
            snapshot = assessor()
            if inspect.isawaitable(snapshot):
                snapshot = await snapshot
        except Exception:  # pragma: no cover - defensive diagnostic guard
            logger.debug("Failed to collect performance health snapshot", exc_info=True)
            return None

        if snapshot is None:
            return None

        if isinstance(snapshot, Mapping):
            items = snapshot.items()
        else:
            try:
                items = dict(snapshot).items()  # type: ignore[arg-type]
            except Exception:
                logger.debug(
                    "Unexpected performance health payload type %s; skipping",
                    type(snapshot).__name__,
                )
                return None

        normalised: dict[str, Any] = {}
        for key, value in items:
            normalised[str(key)] = self._normalise_metadata_value(value)
        return normalised

    @staticmethod
    def _normalise_metadata_value(value: Any) -> Any:
        if isinstance(value, datetime):
            ref = value
            if value.tzinfo is None:
                ref = value.replace(tzinfo=timezone.utc)
            else:
                ref = value.astimezone(timezone.utc)
            return ref.isoformat()
        if isinstance(value, Decimal):
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(key): AlphaTradeLoopRunner._normalise_metadata_value(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [AlphaTradeLoopRunner._normalise_metadata_value(item) for item in value]
        return value

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
