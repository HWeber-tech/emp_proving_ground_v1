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

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import inspect
import logging
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Sequence

from src.operations.drift_sentry import DriftSentryConfig, DriftSentrySnapshot, evaluate_drift_sentry
from src.operations.sensory_drift import DriftSeverity
from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    AlphaTradeLoopResult,
    ComplianceEventType,
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


class OutOfDistributionSentry:
    """Track runtime metrics and surface drift snapshots relative to training."""

    def __init__(
        self,
        *,
        config: DriftSentryConfig | None = None,
        training_window: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._config = config or DriftSentryConfig()
        baseline = max(0, int(self._config.baseline_window))
        evaluation = max(0, int(self._config.evaluation_window))
        self._window = max(1, baseline + evaluation)
        minimum_training = max(self._config.min_observations, 1)
        if training_window is None:
            training_window = max(minimum_training, baseline)
        self._training_cap = max(minimum_training, int(training_window))
        self._history: dict[str, deque[float]] = {}
        self._training_data: dict[str, list[float]] = {}
        self._metadata = dict(metadata or {})
        self._last_snapshot: DriftSentrySnapshot | None = None

    @property
    def config(self) -> DriftSentryConfig:
        return self._config

    @property
    def last_snapshot(self) -> DriftSentrySnapshot | None:
        return self._last_snapshot

    def record(
        self,
        metrics: Mapping[str, float | None],
        *,
        generated_at: datetime | None = None,
    ) -> DriftSentrySnapshot | None:
        if not metrics:
            return None

        for name, raw_value in metrics.items():
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            history = self._history.setdefault(name, deque(maxlen=self._window))
            history.append(value)
            training = self._training_data.setdefault(name, [])
            if len(training) < self._training_cap:
                training.append(value)

        ready_metrics: dict[str, list[float]] = {}
        training_reference: dict[str, list[float]] = {}

        for name, series in self._history.items():
            if len(series) < self._window:
                continue
            training = self._training_data.get(name)
            if not training or len(training) < self._config.min_observations:
                continue
            ready_metrics[name] = list(series)
            training_reference[name] = list(training)

        if not ready_metrics:
            return None

        metadata = dict(self._metadata)
        if training_reference:
            metadata.setdefault("training_ready_metrics", tuple(sorted(training_reference)))

        try:
            snapshot = evaluate_drift_sentry(
                ready_metrics,
                config=self._config,
                generated_at=generated_at,
                metadata=metadata,
                training_reference=training_reference,
            )
        except ValueError:
            return None

        self._last_snapshot = snapshot
        return snapshot


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
    _TOP_FEATURE_NORM_LIMIT = 25.0
    _ATTRIBUTION_EXPLANATION_LIMIT = 160

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
        memory_trust_support: int = 3,
        memory_min_multiplier: float = 0.5,
        ood_sentry: OutOfDistributionSentry | None = None,
        ood_sentry_config: DriftSentryConfig | None = None,
        ood_training_window: int | None = None,
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

        if memory_trust_support < 0:
            raise ValueError("memory_trust_support must be non-negative")
        if not math.isfinite(memory_min_multiplier) or memory_min_multiplier <= 0.0:
            raise ValueError("memory_min_multiplier must be positive")
        if memory_min_multiplier > 1.0:
            raise ValueError("memory_min_multiplier must be <= 1.0")

        self._memory_trust_support = int(memory_trust_support)
        self._memory_min_multiplier = float(memory_min_multiplier)

        sentry_config = ood_sentry_config or DriftSentryConfig()
        self._ood_sentry = ood_sentry or OutOfDistributionSentry(
            config=sentry_config,
            training_window=ood_training_window,
            metadata={"source": "alpha_trade_runner.ood_sentry"},
        )

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
        ood_snapshot: DriftSentrySnapshot | None = None
        if self._ood_sentry is not None:
            ood_metrics = self._collect_out_of_distribution_metrics(
                belief_state=belief_state,
                regime_signal=regime_signal,
                sensory_snapshot=sensory_snapshot if isinstance(sensory_snapshot, Mapping) else None,
            )
            if ood_metrics:
                generated_at = None
                if isinstance(sensory_snapshot, Mapping):
                    generated_at = _coerce_datetime(sensory_snapshot.get("generated_at"))
                if generated_at is None:
                    generated_at = getattr(belief_state, "generated_at", None)
                if isinstance(generated_at, datetime) and generated_at.tzinfo is None:
                    generated_at = generated_at.replace(tzinfo=timezone.utc)
                ood_snapshot = self._ood_sentry.record(
                    ood_metrics,
                    generated_at=generated_at,
                )

        try:
            loop_result = self._orchestrator.run_iteration(
                belief_snapshot,
                belief_state=belief_state,
                policy_id=policy_id,
                trade=trade_plan.metadata,
                notes=tuple(notes or ()),
                extra_metadata=extra_metadata,
                drift_snapshot=ood_snapshot,
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

        existing_guardrails = trade_metadata.get("guardrails")
        raw_guardrails = getattr(loop_result.decision, "guardrails", None)
        resolved_guardrails: dict[str, Any] | None = None
        if isinstance(raw_guardrails, Mapping):
            resolved_guardrails = self._merge_guardrail_payload(
                existing_guardrails,
                raw_guardrails,
            )
        elif isinstance(existing_guardrails, Mapping):
            resolved_guardrails = self._merge_guardrail_payload(None, existing_guardrails)
        if resolved_guardrails is not None:
            trade_metadata["guardrails"] = resolved_guardrails

        diary_entry = loop_result.diary_entry
        has_diary_entry = diary_entry is not None
        attribution_payload = self._build_order_attribution(
            belief_state=belief_state,
            decision_bundle=loop_result.decision_bundle,
            diary_entry=diary_entry,
        )
        brief_explanation_text = ""
        policy_identifier = ""
        diary_entry_identifier = ""
        if attribution_payload:
            trade_metadata["attribution"] = attribution_payload
            explanation_value = (
                attribution_payload.get("brief_explanation")
                if isinstance(attribution_payload, Mapping)
                else None
            )
            if not isinstance(explanation_value, str) or not explanation_value.strip():
                explanation_value = (
                    attribution_payload.get("explanation")
                    if isinstance(attribution_payload, Mapping)
                    else None
                )
            if isinstance(explanation_value, str):
                brief_explanation_text = AlphaTradeLoopRunner._build_brief_explanation(
                    explanation_value
                )
            if isinstance(attribution_payload, Mapping):
                raw_policy_id = attribution_payload.get("policy_id")
                if raw_policy_id:
                    policy_identifier = str(raw_policy_id).strip()
                raw_diary_entry_id = attribution_payload.get("diary_entry_id")
                if raw_diary_entry_id:
                    diary_entry_identifier = str(raw_diary_entry_id).strip()
        else:
            trade_metadata.pop("attribution", None)

        if brief_explanation_text:
            trade_metadata["brief_explanation"] = brief_explanation_text
        else:
            trade_metadata.pop("brief_explanation", None)

        if policy_identifier:
            trade_metadata["policy_id"] = policy_identifier
        else:
            trade_metadata.pop("policy_id", None)

        if diary_entry_identifier:
            trade_metadata["diary_entry_id"] = diary_entry_identifier
        else:
            trade_metadata.pop("diary_entry_id", None)

        intent_payload = None
        if trade_plan.intent is not None:
            raw_intent = dict(trade_plan.intent)
            metadata_payload = dict(raw_intent.get("metadata", {}))
            if "fast_weight" not in metadata_payload:
                metadata_payload["fast_weight"] = dict(fast_weight_metadata)
            if attribution_payload:
                metadata_payload["attribution"] = attribution_payload
            else:
                metadata_payload.pop("attribution", None)

            if brief_explanation_text:
                metadata_payload["brief_explanation"] = brief_explanation_text
            else:
                metadata_payload.pop("brief_explanation", None)

            if policy_identifier:
                metadata_payload["policy_id"] = policy_identifier
            else:
                metadata_payload.pop("policy_id", None)

            if diary_entry_identifier:
                metadata_payload["diary_entry_id"] = diary_entry_identifier
            else:
                metadata_payload.pop("diary_entry_id", None)
            if resolved_guardrails is not None:
                metadata_payload["guardrails"] = self._merge_guardrail_payload(
                    metadata_payload.get("guardrails"),
                    resolved_guardrails,
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

        mitigation_mutable: MutableMapping[str, Any] | None = None
        if isinstance(trade_metadata.get("drift_mitigation"), MutableMapping):
            mitigation_mutable = trade_metadata["drift_mitigation"]  # type: ignore[index]

        theory_packet_mutable: MutableMapping[str, Any] | None = None
        if isinstance(trade_metadata.get("theory_packet"), MutableMapping):
            theory_packet_mutable = trade_metadata["theory_packet"]  # type: ignore[index]

        memory_gate_payload = self._apply_memory_gate(
            belief_state=belief_state,
            trade_metadata=trade_metadata,
            intent_payload=intent_payload,
            mitigation_payload=mitigation_mutable,
            theory_packet_payload=theory_packet_mutable,
        )

        if theory_packet_mutable is not None:
            theory_packet_payload = theory_packet_mutable
        if mitigation_mutable is not None:
            mitigation_payload = mitigation_mutable

        trade_outcome: "TradeIntentOutcome | None" = None
        diary_annotations: dict[str, Any] = {}
        loop_metadata_updates: dict[str, Any] = {}
        if resolved_guardrails is not None:
            guardrail_snapshot = self._merge_guardrail_payload(None, resolved_guardrails)
            loop_metadata_updates["guardrails"] = guardrail_snapshot
            if has_diary_entry:
                diary_annotations["guardrails"] = guardrail_snapshot
        coverage_snapshot = dict(self.describe_diary_coverage())
        loop_metadata_updates["diary_coverage"] = dict(coverage_snapshot)
        trade_metadata["diary_coverage"] = dict(coverage_snapshot)
        if intent_payload is not None:
            if not isinstance(intent_payload, MutableMapping):
                intent_payload = dict(intent_payload)
            intent_metadata = intent_payload.get("metadata")
            if isinstance(intent_metadata, Mapping):
                metadata_payload = dict(intent_metadata)
            else:
                metadata_payload = {}
            metadata_payload["diary_coverage"] = dict(coverage_snapshot)
            intent_payload["metadata"] = metadata_payload
        if has_diary_entry:
            diary_annotations["diary_coverage"] = dict(coverage_snapshot)
        if has_diary_entry and attribution_payload:
            diary_annotations["attribution"] = attribution_payload
        if attribution_payload:
            loop_metadata_updates["attribution"] = attribution_payload
            if brief_explanation_text:
                loop_metadata_updates["brief_explanation"] = brief_explanation_text
            if policy_identifier:
                loop_metadata_updates["policy_id"] = policy_identifier
            if diary_entry_identifier:
                loop_metadata_updates["diary_entry_id"] = diary_entry_identifier
        elif brief_explanation_text:
            loop_metadata_updates["brief_explanation"] = brief_explanation_text
        if memory_gate_payload:
            memory_copy = dict(memory_gate_payload)
            loop_metadata_updates["memory_gate"] = memory_copy
            if has_diary_entry:
                diary_annotations["memory_gate"] = memory_copy
        if mitigation_payload:
            mitigation_copy = dict(mitigation_payload)
            if has_diary_entry:
                diary_annotations["drift_mitigation"] = mitigation_copy
            loop_metadata_updates["drift_mitigation"] = mitigation_copy
        if intent_payload is not None:
            outcome = await self._trading_manager.on_trade_intent(intent_payload)
            if outcome is not None:
                trade_outcome = self._attach_trade_outcome_metadata(
                    trade_outcome=outcome,
                    coverage_snapshot=coverage_snapshot,
                    attribution_payload=attribution_payload,
                    guardrails=resolved_guardrails,
                )
            else:
                trade_outcome = None
            if trade_outcome is not None:
                trade_execution_payload: dict[str, Any] = {
                    "status": trade_outcome.status,
                    "executed": trade_outcome.executed,
                }
                if trade_outcome.metadata:
                    trade_execution_payload["metadata"] = dict(trade_outcome.metadata)
                if trade_outcome.throttle:
                    trade_execution_payload["throttle"] = dict(trade_outcome.throttle)
                if has_diary_entry:
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
                    if has_diary_entry:
                        diary_annotations["drift_throttle"] = throttle_copy
                    loop_metadata_updates["drift_throttle"] = throttle_copy
                if updated_packet is not None:
                    theory_packet_payload = updated_packet

        if theory_packet_payload:
            self._enrich_action_logs(
                theory_packet_payload,
                intent_payload=intent_payload,
                trade_metadata=trade_metadata,
                trade_outcome=trade_outcome,
            )
            trade_metadata["theory_packet"] = dict(theory_packet_payload)
            if isinstance(intent_payload, MutableMapping):
                intent_meta = intent_payload.get("metadata")
                if isinstance(intent_meta, MutableMapping):
                    intent_meta["theory_packet"] = dict(theory_packet_payload)
            packet_copy = dict(theory_packet_payload)
            if has_diary_entry:
                diary_annotations["theory_packet"] = packet_copy
            loop_metadata_updates["theory_packet"] = packet_copy

        performance_health = await self._collect_performance_health()
        if performance_health is not None:
            if has_diary_entry:
                diary_annotations.setdefault("performance_health", performance_health)
            loop_metadata_updates.setdefault("performance_health", performance_health)
            trade_metadata.setdefault("performance_health", performance_health)

        if has_diary_entry and diary_annotations:
            merged_loop_metadata = dict(loop_result.metadata)
            merged_loop_metadata.update(loop_metadata_updates)
            if not brief_explanation_text:
                merged_loop_metadata.pop("brief_explanation", None)
            if not policy_identifier:
                merged_loop_metadata.pop("policy_id", None)
            if not diary_entry_identifier:
                merged_loop_metadata.pop("diary_entry_id", None)
            if attribution_payload is None:
                merged_loop_metadata.pop("attribution", None)
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
            if not brief_explanation_text:
                merged_loop_metadata.pop("brief_explanation", None)
            if not policy_identifier:
                merged_loop_metadata.pop("policy_id", None)
            if not diary_entry_identifier:
                merged_loop_metadata.pop("diary_entry_id", None)
            if attribution_payload is None:
                merged_loop_metadata.pop("attribution", None)
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

    @staticmethod
    def _collect_out_of_distribution_metrics(
        *,
        belief_state: BeliefState,
        regime_signal: RegimeSignal,
        sensory_snapshot: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        belief_metadata = getattr(belief_state, "metadata", None)
        if isinstance(belief_metadata, Mapping):
            confidence_value = _coerce_float(belief_metadata.get("confidence"))
            if confidence_value is not None and math.isfinite(confidence_value):
                metrics["belief_confidence"] = confidence_value

        regime_confidence = _coerce_float(regime_signal.regime_state.confidence)
        if regime_confidence is not None and math.isfinite(regime_confidence):
            metrics["regime_confidence"] = regime_confidence

        if isinstance(sensory_snapshot, Mapping):
            integrated_block = sensory_snapshot.get("integrated_signal")
            if isinstance(integrated_block, Mapping):
                strength_value = _coerce_float(integrated_block.get("strength"))
                if strength_value is not None and math.isfinite(strength_value):
                    metrics["integrated_strength"] = strength_value
                integrated_confidence = _coerce_float(integrated_block.get("confidence"))
                if integrated_confidence is not None and math.isfinite(integrated_confidence):
                    metrics["integrated_confidence"] = integrated_confidence

        return metrics

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
    def _bound_probability(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _bound_feature_norm(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        limit = AlphaTradeLoopRunner._TOP_FEATURE_NORM_LIMIT
        bounded = max(-limit, min(limit, value))
        return bounded

    @staticmethod
    def _normalise_metadata_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, Enum):
            enum_value = value.value
            if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
                return enum_value
            return str(enum_value)
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, Mapping):
            return {
                str(key): AlphaTradeLoopRunner._normalise_metadata_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (set, frozenset)):
            return [
                AlphaTradeLoopRunner._normalise_metadata_value(item)
                for item in sorted(value, key=lambda entry: str(entry))
            ]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [
                AlphaTradeLoopRunner._normalise_metadata_value(item)
                for item in value
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

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
            numeric = _coerce_float(value)
            if numeric is None or not math.isfinite(numeric):
                continue
            bounded = AlphaTradeLoopRunner._bound_feature_norm(numeric)
            ranked.append((str(name), bounded))
        ranked.sort(key=lambda item: abs(item[1]), reverse=True)
        summary: list[Mapping[str, Any]] = []
        for name, value in ranked[:limit]:
            summary.append({"name": name, "value": value})
        return summary

    @staticmethod
    def _build_brief_explanation(value: Any, *, limit: int | None = None) -> str:
        """Collapse whitespace and enforce a maximum explanation length."""

        text = str(value or "").strip()
        if not text:
            return ""

        collapsed = " ".join(text.split())
        resolved_limit = (
            limit
            if isinstance(limit, int) and limit > 0
            else AlphaTradeLoopRunner._ATTRIBUTION_EXPLANATION_LIMIT
        )
        if len(collapsed) <= resolved_limit:
            return collapsed
        if resolved_limit <= 3:
            return collapsed[:resolved_limit]
        return collapsed[: resolved_limit - 3].rstrip() + "..."

    @staticmethod
    def _normalise_probe_fragment(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "unknown"
        cleaned: list[str] = []
        for char in text.lower():
            if char.isalnum() or char in {"-", "_", "."}:
                cleaned.append(char)
            else:
                cleaned.append("_")
        fragment = "".join(cleaned).strip("._-")
        return fragment or "unknown"

    @staticmethod
    def _build_guardrail_probes(guardrails: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(guardrails, Mapping) or not guardrails:
            return []

        probes: list[dict[str, Any]] = []

        counterfactual_payload = guardrails.get("counterfactual_guardrail")
        if isinstance(counterfactual_payload, Mapping):
            breached = any(
                bool(counterfactual_payload.get(flag))
                for flag in ("breached", "relative_breach", "absolute_breach")
            )
            status = "breached" if breached else "ok"
            severity_value = str(counterfactual_payload.get("severity", "")).strip()
            severity = severity_value or ("alert" if breached else "info")
            probe_entry: dict[str, Any] = {
                "probe_id": "guardrail.counterfactual",
                "status": status,
            }
            if severity:
                probe_entry["severity"] = severity
            probes.append(probe_entry)

        force_paper = guardrails.get("force_paper")
        if isinstance(force_paper, bool):
            probe_entry = {
                "probe_id": "guardrail.force_paper",
                "status": "active" if force_paper else "inactive",
            }
            probe_entry["severity"] = "warn" if force_paper else "info"
            probes.append(probe_entry)

        requires_diary = guardrails.get("requires_diary")
        if isinstance(requires_diary, bool):
            probes.append(
                {
                    "probe_id": "guardrail.requires_diary",
                    "status": "active" if requires_diary else "inactive",
                    "severity": "info",
                }
            )

        return probes

    @staticmethod
    def _build_drift_probes(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping) or not payload:
            return []

        allowed = bool(payload.get("allowed"))
        status = "allowed" if allowed else "blocked"
        probe_entry: dict[str, Any] = {
            "probe_id": "drift.sentry",
            "status": status,
        }
        severity_value = payload.get("severity")
        if isinstance(severity_value, str):
            severity_text = severity_value.strip()
            if severity_text:
                probe_entry["severity"] = severity_text
        return [probe_entry]

    def _build_probe_fallbacks(
        self,
        *,
        diary_entry: "DecisionDiaryEntry",
        decision_bundle: UnderstandingDecision,
        belief_state: BeliefState,
    ) -> list[Mapping[str, Any]]:
        metadata = getattr(diary_entry, "metadata", None)

        guardrail_source: Mapping[str, Any] | None = None
        if isinstance(metadata, Mapping):
            guardrail_candidate = metadata.get("guardrails")
            if isinstance(guardrail_candidate, Mapping):
                guardrail_source = guardrail_candidate
        if guardrail_source is None:
            decision_guardrails = getattr(decision_bundle.decision, "guardrails", None)
            if isinstance(decision_guardrails, Mapping):
                guardrail_source = decision_guardrails

        probe_entries = self._build_guardrail_probes(guardrail_source)

        if isinstance(metadata, Mapping):
            drift_candidate = metadata.get("drift_decision")
            probe_entries.extend(self._build_drift_probes(drift_candidate))

        deduplicated: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for entry in probe_entries:
            probe_id = entry.get("probe_id")
            status = entry.get("status")
            if not isinstance(probe_id, str) or not probe_id.strip():
                continue
            if not isinstance(status, str) or not status.strip():
                continue
            key = (probe_id.strip(), status.strip())
            if key in seen:
                continue
            normalised_entry: dict[str, Any] = {
                "probe_id": key[0],
                "status": key[1],
            }
            severity_value = entry.get("severity")
            if isinstance(severity_value, str):
                severity_text = severity_value.strip()
                if severity_text:
                    normalised_entry["severity"] = severity_text
            seen.add(key)
            deduplicated.append(normalised_entry)

        if deduplicated:
            return deduplicated

        policy_fragment = self._normalise_probe_fragment(getattr(diary_entry, "policy_id", ""))
        symbol_fragment = self._normalise_probe_fragment(getattr(belief_state, "symbol", ""))
        fragments = [fragment for fragment in (policy_fragment, symbol_fragment) if fragment and fragment != "unknown"]
        if not fragments:
            fragments = ["attribution"]
        default_probe_id = "probe." + ".".join(fragments)
        return [{"probe_id": default_probe_id, "status": "ok", "severity": "info"}]

    def _build_order_attribution(
        self,
        *,
        belief_state: BeliefState,
        decision_bundle: UnderstandingDecision,
        diary_entry: "DecisionDiaryEntry | None",
    ) -> Mapping[str, Any] | None:
        regime_state = decision_bundle.belief_snapshot.regime_state

        belief_metadata = getattr(belief_state, "metadata", None)
        belief_confidence = (
            _coerce_float(belief_metadata.get("confidence"))
            if isinstance(belief_metadata, Mapping)
            else None
        )

        confidence_value = (
            belief_confidence
            if belief_confidence is not None
            else _coerce_float(regime_state.confidence)
        )

        bounded_confidence: float | None = None
        if confidence_value is not None:
            bounded_confidence = AlphaTradeLoopRunner._bound_probability(
                confidence_value
            )

        belief_id_value = getattr(belief_state, "belief_id", None)
        belief_id = str(belief_id_value).strip() if belief_id_value else ""
        if not belief_id:
            return None

        symbol_value = getattr(belief_state, "symbol", None)
        symbol = str(symbol_value).strip() if symbol_value else ""

        regime_value = getattr(regime_state, "regime", None)
        regime = str(regime_value).strip() if regime_value else ""
        fallback_regime = regime or str(regime_state.regime or "unknown").strip() or "unknown"

        belief_summary: dict[str, Any] = {
            "belief_id": belief_id,
        }

        if bounded_confidence is not None:
            belief_summary["confidence"] = bounded_confidence

        if symbol:
            belief_summary["symbol"] = symbol
        if regime:
            belief_summary["regime"] = regime

        generated_at = belief_state.generated_at
        if isinstance(generated_at, datetime):
            belief_summary["generated_at"] = generated_at.astimezone(timezone.utc).isoformat()

        if isinstance(belief_metadata, Mapping) and belief_metadata:
            metadata_snapshot: dict[str, Any] = {}
            for key, value in belief_metadata.items():
                key_text = str(key)
                if key_text == "confidence":
                    confidence_override = _coerce_float(value)
                    if confidence_override is not None:
                        metadata_snapshot[key_text] = (
                            AlphaTradeLoopRunner._bound_probability(confidence_override)
                        )
                    continue
                normalised_value = AlphaTradeLoopRunner._normalise_metadata_value(value)
                if isinstance(normalised_value, Mapping):
                    normalised_value = dict(normalised_value)
                metadata_snapshot[key_text] = normalised_value
            if metadata_snapshot:
                belief_summary["metadata"] = metadata_snapshot

        top_features = self._select_top_features(decision_bundle.belief_snapshot.features)
        if top_features:
            belief_summary["top_features"] = top_features

        probes_payload: list[Mapping[str, Any]] = []
        probes_source = getattr(diary_entry, "probes", ())
        for activation in probes_source or ():
            probe_id_value = getattr(activation, "probe_id", None)
            status_value = getattr(activation, "status", None)
            probe_id_text = str(probe_id_value).strip() if probe_id_value else ""
            status_text = str(status_value).strip() if status_value else ""
            if not probe_id_text or not status_text:
                continue

            probe_entry: dict[str, Any] = {
                "probe_id": probe_id_text,
                "status": status_text,
            }
            severity_value = getattr(activation, "severity", None)
            severity_text = str(severity_value).strip() if severity_value else ""
            if severity_text:
                probe_entry["severity"] = severity_text
            probes_payload.append(probe_entry)

        if not probes_payload:
            probes_payload.extend(
                self._build_probe_fallbacks(
                    diary_entry=diary_entry,
                    decision_bundle=decision_bundle,
                    belief_state=belief_state,
            )
            )

        brief_explanation = AlphaTradeLoopRunner._build_brief_explanation(
            getattr(decision_bundle.decision, "rationale", "")
        )
        if not brief_explanation:
            tactic_label = str(getattr(decision_bundle.decision, "tactic_id", "") or "").strip()
            if not tactic_label:
                tactic_label = str(getattr(diary_entry, "policy_id", "") or "").strip()
            if not tactic_label:
                tactic_label = symbol or "trade"
            brief_explanation = AlphaTradeLoopRunner._build_brief_explanation(
                f"{tactic_label} routed under {fallback_regime}"
            )
        if not brief_explanation:
            brief_explanation = "Routed under governing regime"

        diary_entry_id = None
        if diary_entry is not None:
            entry_identifier = getattr(diary_entry, "entry_id", None)
            if entry_identifier:
                diary_entry_id = str(entry_identifier).strip()

        policy_identifier = getattr(diary_entry, "policy_id", None) if diary_entry else None
        if not policy_identifier:
            policy_identifier = getattr(decision_bundle.decision, "tactic_id", None)
        policy_id = str(policy_identifier).strip() if policy_identifier else ""

        attribution: dict[str, Any] = {
            "belief": belief_summary,
            "probes": probes_payload,
            "brief_explanation": brief_explanation,
            "explanation": brief_explanation,
        }

        if diary_entry_id:
            attribution["diary_entry_id"] = diary_entry_id
        if policy_id:
            attribution["policy_id"] = policy_id

        return attribution

    def _attach_trade_outcome_metadata(
        self,
        *,
        trade_outcome: "TradeIntentOutcome",
        coverage_snapshot: Mapping[str, Any],
        attribution_payload: Mapping[str, Any] | None,
        guardrails: Mapping[str, Any] | None,
    ) -> "TradeIntentOutcome":
        metadata: dict[str, Any]
        if isinstance(trade_outcome.metadata, Mapping):
            metadata = dict(trade_outcome.metadata)
        else:
            metadata = {}

        changed = False

        brief_explanation_text = ""
        policy_identifier = ""
        diary_entry_identifier = ""
        if isinstance(attribution_payload, Mapping):
            explanation_value = attribution_payload.get("brief_explanation")
            if not isinstance(explanation_value, str) or not explanation_value.strip():
                explanation_value = attribution_payload.get("explanation")
            if isinstance(explanation_value, str):
                brief_explanation_text = AlphaTradeLoopRunner._build_brief_explanation(
                    explanation_value
                )
            raw_policy_id = attribution_payload.get("policy_id")
            if raw_policy_id:
                policy_identifier = str(raw_policy_id).strip()
            raw_diary_entry_id = attribution_payload.get("diary_entry_id")
            if raw_diary_entry_id:
                diary_entry_identifier = str(raw_diary_entry_id).strip()

        coverage_payload = dict(coverage_snapshot)
        if metadata.get("diary_coverage") != coverage_payload:
            metadata["diary_coverage"] = coverage_payload
            changed = True

        if attribution_payload is not None:
            if metadata.get("attribution") != attribution_payload:
                metadata["attribution"] = attribution_payload
                changed = True
        elif metadata.pop("attribution", None) is not None:
            changed = True

        if brief_explanation_text:
            if metadata.get("brief_explanation") != brief_explanation_text:
                metadata["brief_explanation"] = brief_explanation_text
                changed = True
        elif metadata.pop("brief_explanation", None) is not None:
            changed = True

        if policy_identifier:
            if metadata.get("policy_id") != policy_identifier:
                metadata["policy_id"] = policy_identifier
                changed = True
        elif metadata.pop("policy_id", None) is not None:
            changed = True

        if diary_entry_identifier:
            if metadata.get("diary_entry_id") != diary_entry_identifier:
                metadata["diary_entry_id"] = diary_entry_identifier
                changed = True
        elif metadata.pop("diary_entry_id", None) is not None:
            changed = True

        if guardrails is not None:
            existing_guardrails = metadata.get("guardrails")
            if isinstance(existing_guardrails, Mapping):
                combined_guardrails = self._merge_guardrail_payload(
                    existing_guardrails,
                    guardrails,
                )
            else:
                combined_guardrails = self._merge_guardrail_payload(None, guardrails)
            if metadata.get("guardrails") != combined_guardrails:
                metadata["guardrails"] = combined_guardrails
                changed = True

        if not changed:
            return trade_outcome

        return replace(trade_outcome, metadata=metadata)

    def _ensure_action_log_shape(
        self,
        action: MutableMapping[str, Any],
        *,
        default_reason: str | None = None,
        default_context_mult: float | None = None,
    ) -> None:
        reason_value = (
            action.get("reason_code")
            or action.get("reason")
            or default_reason
            or action.get("action")
        )
        if reason_value is not None:
            action["reason_code"] = str(reason_value)
        else:
            action.setdefault("reason_code", None)

        context_value = action.get("context_mult")
        coerced_context = _coerce_float(context_value)
        if coerced_context is not None:
            action["context_mult"] = coerced_context
        else:
            fallback = _coerce_float(action.get("value"))
            if fallback is None and default_context_mult is not None:
                fallback = default_context_mult
            if fallback is not None:
                action["context_mult"] = fallback
            else:
                action.setdefault("context_mult", None)

        for key in ("edge_ticks", "cost_to_take", "inventory", "latency_ms"):
            action.setdefault(key, None)

    def _extract_execution_risk_payload(
        self,
        intent_payload: Mapping[str, Any] | None,
        trade_metadata: Mapping[str, Any] | None,
        trade_outcome: "TradeIntentOutcome | None",
    ) -> Mapping[str, Any] | None:
        def _from_mapping(
            candidate: Mapping[str, Any] | None,
            key: str = "execution_risk",
        ) -> Mapping[str, Any] | None:
            if not isinstance(candidate, Mapping):
                return None
            value = candidate.get(key)
            if isinstance(value, Mapping):
                return dict(value)
            return None

        meta = intent_payload.get("metadata") if isinstance(intent_payload, Mapping) else None
        exec_payload = _from_mapping(meta)
        if exec_payload is not None:
            return exec_payload
        risk_assessment = meta.get("risk_assessment") if isinstance(meta, Mapping) else None
        exec_payload = _from_mapping(risk_assessment, "execution")
        if exec_payload is not None:
            return exec_payload

        exec_payload = _from_mapping(trade_metadata)
        if exec_payload is not None:
            return exec_payload

        outcome_metadata = getattr(trade_outcome, "metadata", None)
        exec_payload = _from_mapping(outcome_metadata)
        if exec_payload is not None:
            return exec_payload

        resolver = getattr(self._trading_manager, "get_last_risk_decision", None)
        if callable(resolver):
            try:
                decision = resolver()
            except Exception:
                decision = None
            if isinstance(decision, Mapping):
                execution_section = decision.get("execution")
                if isinstance(execution_section, Mapping):
                    return dict(execution_section)
        return None

    def _resolve_inventory_snapshot(self) -> Mapping[str, Any] | None:
        monitor = getattr(self._trading_manager, "portfolio_monitor", None)
        if monitor is None:
            return None
        getter = getattr(monitor, "get_state", None)
        if not callable(getter):
            return None
        try:
            state = getter()
        except Exception:
            return None
        if not isinstance(state, Mapping):
            return None
        positions = state.get("open_positions")
        if not isinstance(positions, Mapping):
            return None
        snapshot: dict[str, Any] = {}
        for symbol, payload in positions.items():
            if isinstance(payload, Mapping):
                snapshot[str(symbol)] = {str(key): value for key, value in payload.items()}
            else:
                snapshot[str(symbol)] = payload
        return snapshot

    def _enrich_action_logs(
        self,
        packet: MutableMapping[str, Any] | None,
        *,
        intent_payload: Mapping[str, Any] | None,
        trade_metadata: Mapping[str, Any] | None,
        trade_outcome: "TradeIntentOutcome | None",
    ) -> None:
        if not isinstance(packet, MutableMapping):
            return
        actions = packet.get("actions")
        if not isinstance(actions, list):
            return

        exec_risk = self._extract_execution_risk_payload(
            intent_payload,
            trade_metadata,
            trade_outcome,
        )
        if exec_risk is not None and isinstance(trade_metadata, MutableMapping):
            trade_metadata.setdefault("execution_risk", dict(exec_risk))

        inventory_snapshot = self._resolve_inventory_snapshot()
        latency_ms = None
        if trade_outcome is not None:
            outcome_metadata = getattr(trade_outcome, "metadata", None)
            if isinstance(outcome_metadata, Mapping):
                latency_ms = _coerce_float(outcome_metadata.get("latency_ms"))

        for entry in actions:
            if not isinstance(entry, MutableMapping):
                continue
            self._ensure_action_log_shape(entry)
            if exec_risk is not None:
                entry["edge_ticks"] = _coerce_float(exec_risk.get("edge_ticks"))
                total_cost = exec_risk.get("total_cost_ticks")
                if total_cost is None:
                    total_cost = exec_risk.get("total_cost_bps")
                entry["cost_to_take"] = _coerce_float(total_cost)
            else:
                entry.setdefault("edge_ticks", None)
                entry.setdefault("cost_to_take", None)
            if inventory_snapshot is not None:
                entry["inventory"] = inventory_snapshot
            else:
                entry.setdefault("inventory", None)
            if latency_ms is not None:
                entry["latency_ms"] = latency_ms
            else:
                entry.setdefault("latency_ms", None)

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

        compliance_events = getattr(loop_result, "compliance_events", ())
        for event in compliance_events or ():
            event_type = getattr(event, "event_type", None)
            if isinstance(event_type, Enum):
                event_type_value = event_type.value
            else:
                event_type_value = str(event_type or "")
            event_type_key = event_type_value.strip().lower()
            if event_type_key not in {
                ComplianceEventType.risk_warning.value,
                ComplianceEventType.risk_breach.value,
            }:
                continue

            severity_attr = getattr(event, "severity", None)
            if isinstance(severity_attr, Enum):
                severity_value = severity_attr.value
            else:
                severity_value = str(severity_attr or "")
            severity_label = severity_value.strip().lower()
            if not severity_label:
                severity_label = (
                    "critical"
                    if event_type_key == ComplianceEventType.risk_breach.value
                    else "warn"
                )

            summary = getattr(event, "summary", None)
            reason = summary.strip() if isinstance(summary, str) else ""
            if not reason:
                reason = f"{event_type_key}_observed"

            metadata_payload: dict[str, object] = {}
            event_metadata = getattr(event, "metadata", None)
            if hasattr(event, "as_dict"):
                try:
                    event_dict = dict(event.as_dict())
                except Exception:  # pragma: no cover - defensive guard
                    event_dict = None
                if event_dict:
                    metadata_payload["compliance_event"] = event_dict
            if (
                "compliance_event" not in metadata_payload
                and isinstance(event_metadata, Mapping)
            ):
                metadata_payload["compliance_event"] = {
                    "event_type": event_type_value,
                    "severity": severity_label,
                    "summary": summary,
                    "metadata": dict(event_metadata),
                }
            elif isinstance(event_metadata, Mapping):
                metadata_payload.setdefault("details", dict(event_metadata))

            freeze_triggers.append(
                {
                    "reason": reason,
                    "triggered_by": f"compliance.{event_type_key}",
                    "severity": severity_label,
                    "metadata": metadata_payload,
                }
            )

        guardrail_triggers = self._guardrail_freeze_triggers(
            trade_outcome=trade_outcome,
            loop_result=loop_result,
        )
        if guardrail_triggers:
            freeze_triggers.extend(guardrail_triggers)

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

    def _compute_memory_multiplier(self, support: int) -> tuple[float, bool]:
        """Return the multiplier applied by the memory gate and reinforcement flag."""

        threshold = self._memory_trust_support
        if threshold <= 0:
            return 1.0, True

        support_value = max(0, int(support))
        if support_value >= threshold:
            return 1.0, True

        ratio = support_value / float(threshold)
        multiplier = max(self._memory_min_multiplier, ratio)
        multiplier = min(multiplier, 1.0)
        reinforced = multiplier >= 1.0 - 1e-9
        if reinforced:
            multiplier = 1.0
        return multiplier, reinforced

    def _apply_memory_gate(
        self,
        *,
        belief_state: BeliefState,
        trade_metadata: MutableMapping[str, Any],
        intent_payload: MutableMapping[str, Any] | None,
        mitigation_payload: MutableMapping[str, Any] | None,
        theory_packet_payload: MutableMapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        """Scale trade size when belief support lacks sufficient precedent."""

        posterior = getattr(belief_state, "posterior", None)
        support_raw = getattr(posterior, "support", 0)
        try:
            support = int(support_raw)
        except (TypeError, ValueError):
            support = 0

        multiplier, reinforced = self._compute_memory_multiplier(support)

        applied_at = _coerce_datetime(trade_metadata.get("timestamp"))
        if applied_at is None:
            applied_at = _coerce_datetime(getattr(belief_state, "generated_at", None))
        applied_at = applied_at or datetime.now(tz=timezone.utc)

        threshold = self._memory_trust_support
        memory_payload: dict[str, Any] = {
            "support": support,
            "threshold": threshold,
            "multiplier": multiplier,
            "applied": False,
            "reinforced": reinforced,
            "applied_at": applied_at.astimezone(timezone.utc).isoformat(),
        }

        trade_metadata["memory_support"] = support
        trade_metadata["memory_lighten_multiplier"] = multiplier
        trade_metadata["memory_reinforced"] = reinforced

        intent_meta: MutableMapping[str, Any] | None = None
        if isinstance(intent_payload, MutableMapping):
            meta_block = intent_payload.get("metadata")
            if isinstance(meta_block, MutableMapping):
                intent_meta = meta_block
            elif isinstance(meta_block, Mapping):
                intent_meta = dict(meta_block)
                intent_payload["metadata"] = intent_meta
            else:
                intent_meta = {}
                intent_payload["metadata"] = intent_meta
            intent_meta["memory_support"] = support
            intent_meta["memory_lighten_multiplier"] = multiplier
            intent_meta["memory_reinforced"] = reinforced

        trade_metadata["memory_gate"] = dict(memory_payload)
        if intent_meta is not None:
            intent_meta["memory_gate"] = dict(memory_payload)

        if reinforced or multiplier >= 1.0:
            memory_payload["multiplier"] = 1.0
            trade_metadata["memory_gate"] = dict(memory_payload)
            if intent_meta is not None:
                intent_meta["memory_gate"] = dict(memory_payload)
            return memory_payload

        quantity_value = _coerce_float(trade_metadata.get("quantity"))
        if quantity_value is None and isinstance(intent_payload, Mapping):
            quantity_value = _coerce_float(intent_payload.get("quantity"))

        price_value = _coerce_float(trade_metadata.get("price"))
        if price_value is None and isinstance(intent_payload, Mapping):
            price_value = _coerce_float(intent_payload.get("price"))

        notional_value = _coerce_float(trade_metadata.get("notional"))
        if notional_value is None and isinstance(intent_payload, Mapping):
            intent_meta = intent_payload.get("metadata")
            if isinstance(intent_meta, Mapping):
                notional_value = _coerce_float(intent_meta.get("notional"))

        adjusted_quantity: float | None = None
        adjusted_notional: float | None = None

        if quantity_value is not None:
            adjusted_quantity = quantity_value * multiplier
            trade_metadata["quantity"] = adjusted_quantity
            if isinstance(intent_payload, MutableMapping):
                intent_payload["quantity"] = adjusted_quantity

        original_notional = notional_value
        if original_notional is None and quantity_value is not None and price_value is not None:
            original_notional = abs(quantity_value) * abs(price_value)

        if original_notional is not None:
            adjusted_notional = original_notional * multiplier
            trade_metadata["notional"] = adjusted_notional
            if isinstance(intent_payload, MutableMapping):
                meta_block = intent_payload.get("metadata")
                if isinstance(meta_block, MutableMapping):
                    meta_block["notional"] = adjusted_notional

        applied = (adjusted_quantity is not None) or (adjusted_notional is not None)
        memory_payload.update(
            {
                "applied": applied,
                "original_quantity": quantity_value,
                "adjusted_quantity": adjusted_quantity,
                "original_notional": original_notional,
                "adjusted_notional": adjusted_notional,
            }
        )

        # Update mitigation payloads to reflect final quantities while preserving drift context.
        mitigation_targets: list[MutableMapping[str, Any]] = []
        if isinstance(mitigation_payload, MutableMapping):
            mitigation_targets.append(mitigation_payload)
        drift_block = trade_metadata.get("drift_mitigation")
        if isinstance(drift_block, MutableMapping) and drift_block is not mitigation_payload:
            mitigation_targets.append(drift_block)

        for target in mitigation_targets:
            prior_adjusted_quantity = _coerce_float(target.get("adjusted_quantity"))
            prior_adjusted_notional = _coerce_float(target.get("adjusted_notional"))
            if prior_adjusted_quantity is not None and "drift_adjusted_quantity" not in target:
                target["drift_adjusted_quantity"] = prior_adjusted_quantity
            if prior_adjusted_notional is not None and "drift_adjusted_notional" not in target:
                target["drift_adjusted_notional"] = prior_adjusted_notional
            target["memory_multiplier"] = multiplier
            target["memory_support"] = support
            size_multiplier = _coerce_float(target.get("size_multiplier")) or 1.0
            target["combined_multiplier"] = size_multiplier * multiplier
            if adjusted_quantity is not None:
                target["adjusted_quantity"] = adjusted_quantity
                target["final_quantity"] = adjusted_quantity
            if adjusted_notional is not None:
                target["adjusted_notional"] = adjusted_notional
                target["final_notional"] = adjusted_notional

        if isinstance(theory_packet_payload, MutableMapping):
            actions = theory_packet_payload.get("actions")
            if isinstance(actions, list):
                action_entry: dict[str, Any] = {
                    "action": "memory_multiplier",
                    "value": multiplier,
                    "applied": adjusted_quantity is not None or adjusted_notional is not None,
                    "reason": "memory_support_below_threshold",
                    "reason_code": "memory_support_below_threshold",
                    "context_mult": multiplier,
                }
                if quantity_value is not None:
                    action_entry["original_quantity"] = quantity_value
                if adjusted_quantity is not None:
                    action_entry["adjusted_quantity"] = adjusted_quantity
                if original_notional is not None:
                    action_entry["original_notional"] = original_notional
                if adjusted_notional is not None:
                    action_entry["adjusted_notional"] = adjusted_notional
                self._ensure_action_log_shape(
                    action_entry,
                    default_reason="memory_support_below_threshold",
                    default_context_mult=multiplier,
                )
                actions.append(action_entry)

            summary = str(theory_packet_payload.get("summary", ""))
            addition = (
                f" Memory gate applied {multiplier:.2f}x (support {support}/{threshold})."
            )
            theory_packet_payload["summary"] = (summary + addition).strip()

        trade_metadata["memory_gate"] = dict(memory_payload)

        if intent_meta is not None:
            intent_meta["memory_gate"] = dict(memory_payload)

        return memory_payload

    @staticmethod
    def _extract_size_multiplier_from_actions(actions: Any) -> float | None:
        if isinstance(actions, Sequence):
            for entry in actions:
                if not isinstance(entry, Mapping):
                    continue
                if entry.get("action") != "size_multiplier":
                    continue
                multiplier = _coerce_float(entry.get("value"))
                if multiplier is None:
                    multiplier = _coerce_float(entry.get("context_mult"))
                if multiplier is not None:
                    return max(0.0, min(multiplier, 1.0))
        return None

    def _resolve_drift_size_multiplier(
        self,
        drift_decision: DriftSentryDecision | None,
    ) -> float:
        default_multiplier = 0.5
        if drift_decision is None:
            return default_multiplier

        requirements = drift_decision.requirements
        if isinstance(requirements, Mapping):
            candidate = requirements.get("size_multiplier")
            if candidate is None:
                candidate = requirements.get("recommended_size_multiplier")
            resolved = _coerce_float(candidate)
            if resolved is not None:
                return max(0.0, min(resolved, 1.0))

        metadata = drift_decision.snapshot_metadata
        if isinstance(metadata, Mapping):
            resolved = _coerce_float(metadata.get("recommended_size_multiplier"))
            if resolved is None:
                resolved = self._extract_size_multiplier_from_actions(
                    metadata.get("actions")
                )
            if resolved is not None:
                return max(0.0, min(resolved, 1.0))

        return default_multiplier

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
            if isinstance(packet_payload, MutableMapping):
                actions_block = packet_payload.get("actions")
                if isinstance(actions_block, list):
                    for entry in actions_block:
                        if isinstance(entry, MutableMapping):
                            self._ensure_action_log_shape(entry)
            return dict(existing), packet_payload

        multiplier = self._resolve_drift_size_multiplier(drift_decision)
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
                "reason_code": reason,
                "context_mult": 0.0,
            },
            {
                "action": "size_multiplier",
                "value": multiplier,
                "applied": adjusted_quantity is not None,
                "reason": reason,
                "reason_code": reason,
                "context_mult": multiplier,
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
                    "reason": reason,
                    "reason_code": reason,
                    "context_mult": 1.0,
                }
            )

        self._ensure_action_log_shape(actions[0], default_reason=reason, default_context_mult=0.0)
        self._ensure_action_log_shape(actions[1], default_reason=reason, default_context_mult=multiplier)
        if drift_decision.force_paper:
            self._ensure_action_log_shape(actions[2], default_reason=reason, default_context_mult=1.0)

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

        confidence_override = _coerce_float(metadata.get("confidence"))
        regime_confidence = _coerce_float(regime_signal.regime_state.confidence)
        if confidence_override is not None:
            metadata["confidence"] = self._bound_probability(confidence_override)
        elif regime_confidence is not None:
            metadata["confidence"] = self._bound_probability(regime_confidence)
        else:
            metadata.pop("confidence", None)

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

        intent_confidence = _coerce_float(metadata.get("confidence"))
        if intent_confidence is None and regime_confidence is not None:
            intent_confidence = regime_confidence
        if intent_confidence is not None:
            intent_confidence = self._bound_probability(intent_confidence)
            metadata["confidence"] = intent_confidence
        else:
            intent_confidence = 0.0
            metadata.pop("confidence", None)

        intent: MutableMapping[str, Any] = {
            "strategy_id": metadata.get("policy_id"),
            "symbol": metadata.get("symbol"),
            "side": side_value.upper(),
            "quantity": quantity_value,
            "price": price_value,
            "confidence": intent_confidence,
            "timestamp": intent_timestamp,
            "metadata": {
                "regime": regime_signal.regime_state.regime,
                "confidence": intent_confidence,
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

        coverage_payload = metadata.get("diary_coverage")
        if isinstance(coverage_payload, Mapping):
            intent["metadata"]["diary_coverage"] = {
                str(key): value for key, value in coverage_payload.items()
            }

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
    def _guardrail_freeze_triggers(
        *,
        trade_outcome: "TradeIntentOutcome | None",
        loop_result: AlphaTradeLoopResult,
    ) -> list[dict[str, object]]:
        """Derive exploration freeze triggers from guardrail incidents."""

        sources: list[Mapping[str, Any]] = []
        if trade_outcome is not None:
            metadata = getattr(trade_outcome, "metadata", None)
            if isinstance(metadata, Mapping):
                sources.append(metadata)

        loop_metadata = loop_result.metadata
        if isinstance(loop_metadata, Mapping):
            sources.append(loop_metadata)
            trade_metadata = loop_metadata.get("trade_metadata")
            if isinstance(trade_metadata, Mapping):
                sources.append(trade_metadata)

        triggers: list[dict[str, object]] = []
        seen_incidents: set[str] = set()

        for source in sources:
            guardrails = source.get("guardrails")
            if not isinstance(guardrails, Mapping):
                continue

            for name, payload in guardrails.items():
                if not isinstance(name, str) or not isinstance(payload, Mapping):
                    continue

                incident = payload.get("incident")
                if not isinstance(incident, Mapping):
                    continue

                severity_label = str(
                    incident.get("severity")
                    or payload.get("severity")
                    or ""
                ).strip().lower()
                if severity_label != "violation":
                    continue

                violation_names: list[str] = []
                metadata_block = incident.get("metadata")
                if isinstance(metadata_block, Mapping):
                    raw_violations = metadata_block.get("violations")
                    if isinstance(raw_violations, Sequence):
                        violation_names.extend(
                            str(entry)
                            for entry in raw_violations
                            if entry is not None
                        )

                if not violation_names:
                    checks = incident.get("checks")
                    if isinstance(checks, Sequence):
                        for check in checks:
                            if isinstance(check, Mapping):
                                check_name = check.get("name")
                                if check_name:
                                    violation_names.append(str(check_name))

                if not any("invariant" in violation.lower() for violation in violation_names):
                    continue

                incident_id = incident.get("incident_id")
                if isinstance(incident_id, str):
                    if incident_id in seen_incidents:
                        continue
                    seen_incidents.add(incident_id)

                reason = (
                    incident.get("reason")
                    or incident.get("description")
                    or payload.get("reason")
                    or name
                )
                reason_str = str(reason) if reason is not None else "risk_guardrail_violation"

                trigger_metadata = {
                    "guardrail_name": name,
                    "violations": violation_names,
                    "incident": dict(incident),
                }

                triggers.append(
                    {
                        "reason": reason_str,
                        "triggered_by": "risk_guardrail",
                        "severity": "critical",
                        "metadata": trigger_metadata,
                    }
                )

        return triggers

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

        metadata.setdefault(
            "symbol",
            decision.get("parameters", {}).get("symbol") or belief_state.symbol,
        )
        metadata.setdefault("policy_id", decision.get("tactic_id"))
        metadata.setdefault("side", decision.get("parameters", {}).get("side"))

        confidence_override = _coerce_float(metadata.get("confidence"))
        regime_confidence = _coerce_float(metadata.get("regime_confidence"))
        bounded_confidence = None
        if confidence_override is not None:
            bounded_confidence = AlphaTradeLoopRunner._bound_probability(confidence_override)
        elif regime_confidence is not None:
            bounded_confidence = AlphaTradeLoopRunner._bound_probability(regime_confidence)
        elif isinstance(regime_signal, RegimeSignal):
            fallback_confidence = _coerce_float(regime_signal.regime_state.confidence)
            if fallback_confidence is not None:
                bounded_confidence = AlphaTradeLoopRunner._bound_probability(
                    fallback_confidence
                )

        if bounded_confidence is not None:
            metadata["confidence"] = bounded_confidence
        else:
            metadata.pop("confidence", None)

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

        intent_confidence = bounded_confidence
        if intent_confidence is None:
            fallback_confidence = _coerce_float(regime_signal.regime_state.confidence)
            if fallback_confidence is not None:
                intent_confidence = AlphaTradeLoopRunner._bound_probability(
                    fallback_confidence
                )
        if intent_confidence is None:
            intent_confidence = 0.0

        intent_metadata: MutableMapping[str, Any] = {
            "regime": regime_signal.regime_state.regime,
            "confidence": intent_confidence,
        }

        for key in ("brief_explanation", "policy_id", "diary_entry_id"):
            value = metadata.get(key)
            if value:
                intent_metadata[key] = value

        def _attach_metadata_block(key: str) -> None:
            value = metadata.get(key)
            if isinstance(value, Mapping):
                intent_metadata[key] = dict(value)

        for entry_key in (
            "fast_weight",
            "guardrails",
            "attribution",
            "feature_flags",
            "performance_health",
            "diary_coverage",
        ):
            _attach_metadata_block(entry_key)

        intent: MutableMapping[str, Any] = {
            "strategy_id": metadata.get("policy_id") or decision.get("tactic_id"),
            "symbol": metadata.get("symbol"),
            "side": side,
            "quantity": quantity,
            "price": price,
            "confidence": intent_confidence,
            "timestamp": timestamp_parsed or belief_state.generated_at or datetime.now(),
            "metadata": intent_metadata,
        }

        notional = _coerce_float(metadata.get("notional"))
        if notional is not None:
            intent_metadata["notional"] = notional

        ticket = metadata.get("ticket")
        if ticket:
            intent["ticket"] = ticket

        release_stage = metadata.get("release_stage")
        if release_stage:
            intent_metadata["release_stage"] = release_stage

        coverage_payload = intent_metadata.get("diary_coverage")
        if isinstance(coverage_payload, Mapping):
            intent_metadata["diary_coverage"] = {
                str(key): value for key, value in coverage_payload.items()
            }

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
            "reason": throttle_payload.get("reason") or "alpha_decay",
            "reason_code": throttle_payload.get("reason") or "alpha_decay",
            "context_mult": multiplier,
        }
        if quantity_before is not None:
            action_entry["quantity_before"] = quantity_before
        if quantity_after is not None:
            action_entry["quantity_after"] = quantity_after
        if isinstance(throttle_snapshot, Mapping):
            action_entry["throttle_state"] = throttle_snapshot.get("state")

        self._ensure_action_log_shape(
            action_entry,
            default_reason=throttle_payload.get("reason") or "alpha_decay",
            default_context_mult=multiplier,
        )

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

        self._enrich_action_logs(
            packet,
            intent_payload=intent_payload,
            trade_metadata=trade_metadata,
            trade_outcome=trade_outcome,
        )

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
