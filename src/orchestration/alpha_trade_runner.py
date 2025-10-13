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
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import inspect
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Sequence

from src.operations.sensory_drift import DriftSeverity
from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    AlphaTradeLoopResult,
)
from src.thinking.adaptation.feature_toggles import AdaptationFeatureToggles
from src.understanding.belief import BeliefEmitter, BeliefState, RegimeFSM, RegimeSignal
from src.understanding.router import BeliefSnapshot, UnderstandingRouter
from src.trading.gating import serialise_drift_decision

if TYPE_CHECKING:
    from src.trading.trading_manager import TradeIntentOutcome
    from src.understanding.decision_diary import DecisionDiaryEntry


logger = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
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

    _DEFAULT_EXPLORATION_RELEASE_THRESHOLD = 3
    _RISK_REJECTION_STATUS = "rejected"
    _SEVERITY_ORDER = {
        "critical": 3,
        "alert": 2,
        "warn": 1,
        "info": 0,
    }

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
        feature_toggles: AdaptationFeatureToggles | None = None,
        diary_coverage_target: float = 0.95,
        diary_minimum_samples: int = 20,
        diary_gap_alert: timedelta = timedelta(minutes=5),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._belief_emitter = belief_emitter
        self._regime_fsm = regime_fsm
        self._orchestrator = orchestrator
        self._trading_manager = trading_manager
        self._understanding_router = understanding_router
        self._publish_regime_signal = publish_regime_signal
        self._trade_builder = trade_builder or self._default_trade_builder
        self._feature_toggles = feature_toggles or AdaptationFeatureToggles()
        self._exploration_safe_streak = 0
        self._exploration_release_threshold = self._DEFAULT_EXPLORATION_RELEASE_THRESHOLD
        self._last_freeze_reason: str | None = None
        self._diary_target = max(0.0, min(1.0, float(diary_coverage_target)))
        self._diary_min_samples = max(0, int(diary_minimum_samples))
        self._diary_gap_alert = diary_gap_alert if diary_gap_alert >= timedelta(0) else timedelta(0)
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._diary_iterations = 0
        self._diary_recorded = 0
        self._diary_missing = 0
        self._diary_coverage_value = 1.0
        self._diary_alert_emitted = False
        self._diary_gap_alert_emitted = False
        self._last_diary_recorded_at: datetime | None = None

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

        snapshot_flags = None
        flags_candidate = (
            sensory_snapshot.get("feature_flags") if isinstance(sensory_snapshot, Mapping) else None
        )
        if isinstance(flags_candidate, Mapping):
            snapshot_flags = {str(key): bool(value) for key, value in flags_candidate.items()}

        feature_flags = self._feature_toggles.merge_flags(snapshot_flags)
        if not feature_flags:
            feature_flags = None

        fast_weights_enabled_hint: bool | None = None
        fast_flag = sensory_snapshot.get("fast_weights_enabled") if isinstance(sensory_snapshot, Mapping) else None
        if isinstance(fast_flag, bool):
            fast_weights_enabled_hint = fast_flag

        fast_weights_enabled = self._feature_toggles.resolve_fast_weights_enabled(
            fast_weights_enabled_hint
        )

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

        recorded_at: datetime | None = None
        try:
            loop_result = self._orchestrator.run_iteration(
                belief_snapshot,
                belief_state=belief_state,
                policy_id=policy_id,
                trade=trade_plan.metadata,
                notes=tuple(notes or ()),
                extra_metadata=extra_metadata,
            )
        except Exception:
            self._record_diary_result(success=False, recorded_at=None)
            raise
        else:
            diary_entry = getattr(loop_result, "diary_entry", None)
            if diary_entry is not None:
                recorded_at = getattr(diary_entry, "recorded_at", None)
            self._record_diary_result(success=diary_entry is not None, recorded_at=recorded_at)

        fast_weight_metadata = self._build_fast_weight_metadata(
            loop_result.decision_bundle,
            belief_snapshot,
        )

        trade_metadata = dict(trade_plan.metadata or {})
        if "fast_weight" not in trade_metadata:
            trade_metadata["fast_weight"] = dict(fast_weight_metadata)
        guardrails_payload = dict(loop_result.decision.guardrails)
        merged_guardrails: dict[str, Any] | None = None
        if guardrails_payload:
            merged_guardrails = self._merge_guardrail_payload(
                trade_metadata.get("guardrails"),
                guardrails_payload,
            )
            trade_metadata["guardrails"] = merged_guardrails

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
            if guardrails_payload:
                base_guardrails = metadata_payload.get("guardrails")
                if base_guardrails is None and merged_guardrails is not None:
                    base_guardrails = merged_guardrails
                metadata_payload["guardrails"] = self._merge_guardrail_payload(
                    base_guardrails,
                    guardrails_payload,
                )
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

        mitigation_payload: Mapping[str, Any] | None
        theory_packet_payload: Mapping[str, Any] | None
        mitigation_payload, theory_packet_payload = self._apply_drift_mitigation(
            drift_decision=loop_result.drift_decision,
            trade_metadata=trade_metadata,
            intent_payload=intent_payload,
        )

        trade_outcome: "TradeIntentOutcome | None" = None
        diary_annotations: dict[str, Any] = {}
        loop_metadata_updates: dict[str, Any] = {}
        coverage_snapshot = dict(self.describe_diary_coverage())
        loop_metadata_updates.setdefault("diary_coverage", coverage_snapshot)
        trade_metadata.setdefault("diary_coverage", dict(coverage_snapshot))
        if attribution_payload:
            diary_annotations["attribution"] = attribution_payload
            loop_metadata_updates["attribution"] = attribution_payload
        if mitigation_payload:
            mitigation_copy = dict(mitigation_payload)
            diary_annotations["drift_mitigation"] = mitigation_copy
            loop_metadata_updates["drift_mitigation"] = mitigation_copy
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

                throttle_payload, updated_packet = self._apply_throttle_decay(
                    trade_outcome=trade_outcome,
                    trade_metadata=trade_metadata,
                    intent_payload=intent_payload,
                    existing_theory_packet=theory_packet_payload,
                )
                if throttle_payload:
                    throttle_copy = dict(throttle_payload)
                    diary_annotations["drift_throttle"] = throttle_copy
                    loop_metadata_updates["drift_throttle"] = throttle_copy
                if updated_packet is not None:
                    theory_packet_payload = updated_packet

        if theory_packet_payload:
            packet_copy = dict(theory_packet_payload)
            diary_annotations["theory_packet"] = packet_copy
            loop_metadata_updates["theory_packet"] = packet_copy

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
        elif loop_metadata_updates:
            merged_loop_metadata = dict(loop_result.metadata)
            merged_loop_metadata.update(loop_metadata_updates)
            loop_result = replace(
                loop_result,
                metadata=MappingProxyType(merged_loop_metadata),
            )

        if loop_result.metadata.get("trade_metadata") != trade_metadata:
            merged_loop_metadata = dict(loop_result.metadata)
            merged_loop_metadata["trade_metadata"] = dict(trade_metadata)
            loop_result = replace(
                loop_result,
                metadata=MappingProxyType(merged_loop_metadata),
            )

        self._handle_exploration_freeze(
            loop_result=loop_result,
            trade_outcome=trade_outcome,
        )

        return AlphaTradeRunResult(
            belief_state=belief_state,
            regime_signal=regime_signal,
            loop_result=loop_result,
            trade_metadata=dict(trade_metadata),
            trade_intent=dict(intent_payload) if intent_payload is not None else None,
            trade_outcome=trade_outcome,
        )

    def describe_diary_coverage(self) -> Mapping[str, Any]:
        snapshot: dict[str, Any] = {
            "target": self._diary_target,
            "iterations": self._diary_iterations,
            "recorded": self._diary_recorded,
            "missing": self._diary_missing,
            "coverage": round(self._diary_coverage_value, 4),
            "minimum_samples": self._diary_min_samples,
        }
        if self._last_diary_recorded_at is not None:
            snapshot["last_recorded_at"] = self._last_diary_recorded_at.isoformat()
        if self._diary_gap_alert > timedelta(0):
            snapshot["gap_threshold_seconds"] = self._diary_gap_alert.total_seconds()
            snapshot["gap_breach"] = self._diary_gap_alert_emitted
        return snapshot

    def _record_diary_result(self, *, success: bool, recorded_at: datetime | None) -> None:
        now = self._current_time()
        previous_last = self._last_diary_recorded_at

        self._diary_iterations += 1

        if success:
            resolved_recorded = self._ensure_utc(recorded_at) if recorded_at is not None else now
            self._diary_recorded += 1
            if previous_last is None or resolved_recorded >= previous_last:
                self._last_diary_recorded_at = resolved_recorded
            else:
                self._last_diary_recorded_at = previous_last
        else:
            self._diary_missing += 1

        if self._diary_iterations:
            self._diary_coverage_value = self._diary_recorded / self._diary_iterations
        else:  # pragma: no cover - defensive guard
            self._diary_coverage_value = 1.0

        current_last = self._last_diary_recorded_at

        self._evaluate_diary_coverage()
        self._evaluate_diary_gap(
            now=now,
            previous_recorded=previous_last,
            current_recorded=current_last,
            success=success,
        )

    def _evaluate_diary_coverage(self) -> None:
        if self._diary_iterations < self._diary_min_samples:
            return
        target = self._diary_target
        if target <= 0.0:
            return
        coverage = self._diary_coverage_value
        if coverage + 1e-9 < target:
            if not self._diary_alert_emitted:
                logger.warning(
                    "Decision diary coverage %.2f%% below %.2f%% target",
                    coverage * 100.0,
                    target * 100.0,
                    extra={
                        "diary_iterations": self._diary_iterations,
                        "diary_recorded": self._diary_recorded,
                        "diary_missing": self._diary_missing,
                    },
                )
                self._diary_alert_emitted = True
        elif coverage >= target and self._diary_alert_emitted:
            logger.info(
                "Decision diary coverage recovered to %.2f%% (target %.2f%%)",
                coverage * 100.0,
                target * 100.0,
            )
            self._diary_alert_emitted = False

    def _evaluate_diary_gap(
        self,
        *,
        now: datetime,
        previous_recorded: datetime | None,
        current_recorded: datetime | None,
        success: bool,
    ) -> None:
        threshold = self._diary_gap_alert
        if threshold <= timedelta(0):
            return
        if not success:
            reference = previous_recorded
            if reference is None:
                return
            elapsed = now - reference
            if elapsed <= timedelta(0):
                return
            if elapsed > threshold and not self._diary_gap_alert_emitted:
                logger.warning(
                    "Decision diary entries stale for %.1fs (threshold %.1fs)",
                    elapsed.total_seconds(),
                    threshold.total_seconds(),
                    extra={
                        "diary_gap_seconds": elapsed.total_seconds(),
                        "diary_last_recorded_at": reference.isoformat(),
                    },
                )
                self._diary_gap_alert_emitted = True
            return

        if success:
            if self._diary_gap_alert_emitted and current_recorded is not None:
                elapsed_since_recorded = now - current_recorded
                if elapsed_since_recorded < timedelta(0):
                    elapsed_since_recorded = timedelta(0)
                logger.info(
                    "Decision diary entries refreshed after %.1fs gap (threshold %.1fs)",
                    elapsed_since_recorded.total_seconds(),
                    threshold.total_seconds(),
                )
            self._diary_gap_alert_emitted = False

    def _current_time(self) -> datetime:
        candidate = self._clock()
        if isinstance(candidate, datetime):
            return candidate if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)
        raise TypeError("clock must return datetime instances")

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

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

    def _handle_exploration_freeze(
        self,
        *,
        loop_result: AlphaTradeLoopResult,
        trade_outcome: "TradeIntentOutcome | None",
    ) -> None:
        """Apply or release exploration freezes based on loop outcomes."""

        policy_router = self._understanding_router.policy_router

        drift_trigger_metadata: dict[str, object] | None = None
        drift_decision = loop_result.drift_decision
        freeze_triggers: list[dict[str, object]] = []
        if drift_decision is not None:
            drift_snapshot = drift_decision.as_dict()
            drift_trigger_metadata = {"drift_decision": dict(drift_snapshot)}
            severity = drift_decision.severity
            drift_severity_value = severity.value if isinstance(severity, DriftSeverity) else None
            stage_gate = False
            reason_text = drift_decision.reason or ""
            if isinstance(reason_text, str) and reason_text.startswith("release_stage_"):
                stage_gate = True
            if (
                severity in (DriftSeverity.warn, DriftSeverity.alert)
                or not drift_decision.allowed
                or (drift_decision.force_paper and not stage_gate)
            ):
                reason = drift_decision.reason or f"drift_{drift_severity_value or 'unknown'}"
                freeze_triggers.append(
                    {
                        "reason": reason,
                        "triggered_by": "drift_sentry",
                        "severity": drift_severity_value or "warn",
                        "metadata": drift_trigger_metadata,
                    }
                )

        risk_trigger_metadata: dict[str, object] | None = None
        if trade_outcome is not None:
            status = str(getattr(trade_outcome, "status", "")).lower()
            if status == self._RISK_REJECTION_STATUS:
                outcome_payload = trade_outcome.as_dict()
                risk_trigger_metadata = {"trade_outcome": dict(outcome_payload)}
                raw_reason = None
                metadata = getattr(trade_outcome, "metadata", None)
                if isinstance(metadata, Mapping):
                    raw_reason = metadata.get("reason")
                reason_text = str(raw_reason).strip() if raw_reason else "risk_rejected"
                freeze_triggers.append(
                    {
                        "reason": reason_text,
                        "triggered_by": "risk_gateway",
                        "severity": "critical",
                        "metadata": risk_trigger_metadata,
                    }
                )

        if freeze_triggers:
            def _severity_rank(entry: Mapping[str, object]) -> int:
                label = str(entry.get("severity") or "info").lower()
                return self._SEVERITY_ORDER.get(label, 0)

            primary = max(freeze_triggers, key=_severity_rank)
            freeze_metadata: dict[str, object] = {
                "triggers": [
                    {
                        "reason": trigger.get("reason"),
                        "triggered_by": trigger.get("triggered_by"),
                        "severity": trigger.get("severity"),
                        "metadata": (
                            dict(trigger_metadata)
                            if isinstance((trigger_metadata := trigger.get("metadata")), Mapping)
                            else {}
                        ),
                    }
                    for trigger in freeze_triggers
                ]
            }
            policy_router.freeze_exploration(
                reason=str(primary.get("reason") or "exploration_freeze"),
                triggered_by=str(primary.get("triggered_by") or "unknown"),
                severity=str(primary.get("severity")) if primary.get("severity") else None,
                metadata=freeze_metadata,
            )
            self._exploration_safe_streak = 0
            self._last_freeze_reason = str(primary.get("reason") or "exploration_freeze")
            return

        if not policy_router.exploration_freeze_active():
            self._exploration_safe_streak = 0
            return

        drift_safe = True
        if drift_decision is not None:
            severity = drift_decision.severity
            stage_gate = False
            reason_text = drift_decision.reason or ""
            if isinstance(reason_text, str) and reason_text.startswith("release_stage_"):
                stage_gate = True
            drift_safe = (
                severity is DriftSeverity.normal
                and drift_decision.allowed
                and (not drift_decision.force_paper or stage_gate)
            )

        trade_safe = bool(trade_outcome and getattr(trade_outcome, "executed", False))

        stage_gate = False
        if drift_decision is not None:
            reason_text = drift_decision.reason or ""
            if isinstance(reason_text, str) and reason_text.startswith("release_stage_"):
                stage_gate = True

        if drift_safe and trade_safe:
            self._exploration_safe_streak += 1
        else:
            self._exploration_safe_streak = 0

        if self._exploration_safe_streak >= self._exploration_release_threshold:
            policy_router.release_exploration(
                reason="stability_recovered",
                metadata={
                    "safe_iterations": self._exploration_safe_streak,
                    "last_freeze_reason": self._last_freeze_reason,
                    "stage_gate": stage_gate,
                },
            )
            self._exploration_safe_streak = 0
            self._last_freeze_reason = None

    def _apply_drift_mitigation(
        self,
        *,
        drift_decision: DriftSentryDecision | None,
        trade_metadata: MutableMapping[str, Any],
        intent_payload: MutableMapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        """Apply drift-driven mitigations (size reduction & theory packet)."""

        if drift_decision is None:
            return None, None

        severity = drift_decision.severity
        if severity not in (DriftSeverity.warn, DriftSeverity.alert):
            return None, None

        if not isinstance(trade_metadata, MutableMapping):
            return None, None

        existing = trade_metadata.get("drift_mitigation")
        if isinstance(existing, Mapping) and existing.get("size_multiplier") is not None:
            theory_packet = trade_metadata.get("theory_packet")
            packet_payload = (
                dict(theory_packet) if isinstance(theory_packet, Mapping) else None
            )
            return dict(existing), packet_payload

        multiplier = 0.5
        applied_at = _coerce_datetime(trade_metadata.get("timestamp"))
        if applied_at is None and isinstance(intent_payload, Mapping):
            applied_at = _coerce_datetime(intent_payload.get("timestamp"))
        applied_at = applied_at or datetime.now(timezone.utc)
        applied_iso = applied_at.astimezone(timezone.utc).isoformat()
        reason = drift_decision.reason or f"drift_{severity.value}"

        quantity_value = _coerce_float(trade_metadata.get("quantity"))
        if quantity_value is None and isinstance(intent_payload, Mapping):
            quantity_value = _coerce_float(intent_payload.get("quantity"))

        notional_value = _coerce_float(trade_metadata.get("notional"))
        if notional_value is None and isinstance(intent_payload, Mapping):
            metadata_block = intent_payload.get("metadata")
            if isinstance(metadata_block, Mapping):
                notional_value = _coerce_float(metadata_block.get("notional"))

        price_value = _coerce_float(trade_metadata.get("price"))
        if notional_value is None and quantity_value is not None and price_value is not None:
            notional_value = abs(quantity_value) * abs(price_value)

        adjusted_quantity = (
            quantity_value * multiplier if quantity_value is not None else None
        )
        adjusted_notional = (
            notional_value * multiplier if notional_value is not None else None
        )

        if adjusted_quantity is not None:
            trade_metadata["quantity"] = adjusted_quantity
        if adjusted_notional is not None:
            trade_metadata["notional"] = adjusted_notional

        mitigation_payload: dict[str, Any] = {
            "severity": severity.value,
            "reason": reason,
            "size_multiplier": multiplier,
            "applied_at": applied_iso,
            "force_paper": drift_decision.force_paper,
        }
        if isinstance(drift_decision.requirements, Mapping):
            mitigation_payload["requirements"] = dict(drift_decision.requirements)
        if quantity_value is not None:
            mitigation_payload["original_quantity"] = quantity_value
        if adjusted_quantity is not None:
            mitigation_payload["adjusted_quantity"] = adjusted_quantity
        if notional_value is not None:
            mitigation_payload["original_notional"] = notional_value
        if adjusted_notional is not None:
            mitigation_payload["adjusted_notional"] = adjusted_notional

        trade_metadata["drift_mitigation"] = dict(mitigation_payload)

        if isinstance(intent_payload, MutableMapping):
            intent_payload["quantity"] = (
                adjusted_quantity if adjusted_quantity is not None else intent_payload.get("quantity")
            )
            meta_block: MutableMapping[str, Any]
            metadata_payload = intent_payload.get("metadata")
            if isinstance(metadata_payload, MutableMapping):
                meta_block = metadata_payload
            elif isinstance(metadata_payload, Mapping):
                meta_block = dict(metadata_payload)
                intent_payload["metadata"] = meta_block
            else:
                meta_block = {}
                intent_payload["metadata"] = meta_block
            if adjusted_notional is not None:
                meta_block["notional"] = adjusted_notional
            meta_block["drift_mitigation"] = dict(mitigation_payload)

        snapshot_metadata: dict[str, Any] = {}
        if isinstance(drift_decision.snapshot_metadata, Mapping):
            snapshot_metadata = dict(drift_decision.snapshot_metadata)

        actions: list[dict[str, Any]] = [
            {
                "action": "freeze_exploration",
                "status": "triggered",
                "reason": reason,
            },
            {
                "action": "size_multiplier",
                "value": multiplier,
                "applied": adjusted_quantity is not None,
            },
        ]
        if adjusted_quantity is not None:
            actions[1]["original_quantity"] = quantity_value
            actions[1]["adjusted_quantity"] = adjusted_quantity
        if adjusted_notional is not None:
            actions[1]["original_notional"] = notional_value
            actions[1]["adjusted_notional"] = adjusted_notional
        if drift_decision.force_paper:
            actions.append(
                {
                    "action": "force_paper",
                    "status": bool(drift_decision.force_paper),
                }
            )

        drift_snapshot = serialise_drift_decision(drift_decision, evaluated_at=applied_at)

        theory_packet: dict[str, Any] = {
            "summary": (
                f"Drift sentry severity {severity.value} triggered exploration freeze and "
                f"a {multiplier:.2f}x size multiplier"
            ),
            "generated_at": applied_iso,
            "severity": severity.value,
            "actions": actions,
            "drift_decision": drift_snapshot,
        }
        if snapshot_metadata:
            theory_packet["snapshot_metadata"] = snapshot_metadata
            runbook = snapshot_metadata.get("runbook")
            if isinstance(runbook, str) and runbook.strip():
                theory_packet["runbook"] = runbook.strip()

        trade_metadata["theory_packet"] = dict(theory_packet)
        if isinstance(intent_payload, MutableMapping):
            meta_block = intent_payload.get("metadata")
            if isinstance(meta_block, MutableMapping):
                meta_block["theory_packet"] = dict(theory_packet)

        return dict(mitigation_payload), dict(theory_packet)

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
    def _merge_guardrail_payload(
        existing: Mapping[str, Any] | None,
        addition: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Merge guardrail payloads without mutating the original mappings."""

        base: dict[str, Any] = {}
        if isinstance(existing, Mapping):
            for key, value in existing.items():
                base[str(key)] = AlphaTradeLoopRunner._clone_guardrail_value(value)

        for key, value in addition.items():
            key_str = str(key)
            current = base.get(key_str)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                base[key_str] = AlphaTradeLoopRunner._merge_guardrail_payload(current, value)
                continue
            base[key_str] = AlphaTradeLoopRunner._clone_guardrail_value(value)

        return base

    @staticmethod
    def _clone_guardrail_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): AlphaTradeLoopRunner._clone_guardrail_value(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [AlphaTradeLoopRunner._clone_guardrail_value(item) for item in value]
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

    def _apply_throttle_decay(
        self,
        *,
        trade_outcome: "TradeIntentOutcome | None",
        trade_metadata: MutableMapping[str, Any],
        intent_payload: MutableMapping[str, Any] | None,
        existing_theory_packet: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        """Capture trade throttle alpha-decay metadata and theory packet."""

        if trade_outcome is None:
            return None, existing_theory_packet

        outcome_metadata = getattr(trade_outcome, "metadata", None)
        if not isinstance(outcome_metadata, Mapping):
            return None, existing_theory_packet

        multiplier = _coerce_float(
            outcome_metadata.get("throttle_multiplier")
            or outcome_metadata.get("multiplier")
        )
        if multiplier is None or multiplier <= 0.0 or abs(multiplier - 1.0) <= 1e-9:
            return None, existing_theory_packet

        quantity_before = _coerce_float(
            outcome_metadata.get("quantity_before_throttle")
            or outcome_metadata.get("quantity_before")
        )
        quantity_after = _coerce_float(
            outcome_metadata.get("quantity_after_throttle")
            or outcome_metadata.get("quantity_after")
            or outcome_metadata.get("quantity")
        )

        if quantity_after is not None:
            trade_metadata["quantity"] = quantity_after

        notional_value = _coerce_float(outcome_metadata.get("notional"))
        if notional_value is None and quantity_after is not None:
            price_hint = _coerce_float(outcome_metadata.get("price"))
            if price_hint is None:
                price_hint = _coerce_float(trade_metadata.get("price"))
            if price_hint is not None:
                notional_value = abs(quantity_after) * abs(price_hint)
        if notional_value is not None:
            trade_metadata["notional"] = notional_value

        applied_reference = _coerce_datetime(trade_metadata.get("timestamp"))
        applied_reference = applied_reference or datetime.now(timezone.utc)
        applied_iso = applied_reference.astimezone(timezone.utc).isoformat()

        throttle_snapshot = getattr(trade_outcome, "throttle", None)
        throttle_payload: dict[str, Any] = {
            "multiplier": multiplier,
            "applied_at": applied_iso,
            "status": str(getattr(trade_outcome, "status", "executed")),
            "reason": outcome_metadata.get("reason") or "alpha_decay",
            "source": "trade_throttle",
        }
        if quantity_before is not None:
            throttle_payload["quantity_before"] = quantity_before
        if quantity_after is not None:
            throttle_payload["quantity_after"] = quantity_after
        if notional_value is not None:
            throttle_payload["notional"] = notional_value
        if isinstance(throttle_snapshot, Mapping):
            throttle_payload["snapshot"] = dict(throttle_snapshot)

        trade_metadata["drift_throttle"] = dict(throttle_payload)

        if isinstance(intent_payload, MutableMapping):
            if quantity_after is not None:
                intent_payload["quantity"] = quantity_after
            intent_meta = intent_payload.get("metadata")
            if isinstance(intent_meta, MutableMapping):
                if notional_value is not None:
                    intent_meta["notional"] = notional_value
                intent_meta["drift_throttle"] = dict(throttle_payload)
            elif isinstance(intent_meta, Mapping):
                merged_meta = dict(intent_meta)
                if notional_value is not None:
                    merged_meta["notional"] = notional_value
                merged_meta["drift_throttle"] = dict(throttle_payload)
                intent_payload["metadata"] = merged_meta
            else:
                payload_meta: dict[str, Any] = {"drift_throttle": dict(throttle_payload)}
                if notional_value is not None:
                    payload_meta["notional"] = notional_value
                intent_payload["metadata"] = payload_meta

        action_entry: dict[str, Any] = {
            "action": "alpha_decay",
            "value": multiplier,
            "applied_at": applied_iso,
            "status": throttle_payload["status"],
        }
        if quantity_before is not None:
            action_entry["quantity_before"] = quantity_before
        if quantity_after is not None:
            action_entry["quantity_after"] = quantity_after
        if isinstance(throttle_snapshot, Mapping):
            action_entry["throttle_state"] = throttle_snapshot.get("state")

        if existing_theory_packet is not None:
            packet = dict(existing_theory_packet)
            actions = list(packet.get("actions", []))
            actions.append(action_entry)
            packet["actions"] = actions
            summary = str(packet.get("summary") or "").strip()
            addition = f"alpha decay multiplier {multiplier:.2f} via trade throttle"
            if addition.lower() not in summary.lower():
                packet["summary"] = f"{summary + '; ' if summary else ''}{addition}"
            packet.setdefault("generated_at", applied_iso)
            packet.setdefault("severity", "info")
        else:
            packet = {
                "summary": f"Trade throttle applied alpha decay multiplier {multiplier:.2f}",
                "generated_at": applied_iso,
                "severity": "info",
                "actions": [action_entry],
            }

        packet["alpha_decay_multiplier"] = multiplier
        packet["status"] = throttle_payload["status"]
        packet.setdefault("reason", throttle_payload.get("reason"))
        if isinstance(throttle_snapshot, Mapping):
            packet["throttle_snapshot"] = dict(throttle_snapshot)

        trade_metadata["theory_packet"] = dict(packet)

        if isinstance(intent_payload, MutableMapping):
            intent_meta = intent_payload.get("metadata")
            if isinstance(intent_meta, MutableMapping):
                intent_meta["theory_packet"] = dict(packet)
            elif isinstance(intent_meta, Mapping):
                merged_meta = dict(intent_meta)
                merged_meta["theory_packet"] = dict(packet)
                intent_payload["metadata"] = merged_meta
            else:
                intent_payload["metadata"] = {
                    "theory_packet": dict(packet),
                }

        return throttle_payload, packet


__all__ = [
    "AlphaTradeLoopRunner",
    "AlphaTradeRunResult",
    "TradePlan",
]
