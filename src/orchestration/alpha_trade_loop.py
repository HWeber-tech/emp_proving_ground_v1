"""AlphaTrade loop orchestrator for tactic experimentation and staged promotions.

This module implements the roadmap deliverable that expands the AlphaTrade
Perception → Adaptation → Reflection loop beyond the live-shadow pilot.  It
bridges the understanding router (fast-weight experiments), DriftSentry gating,
policy ledger release stages, and the decision diary so tactic experimentation
feeds paper-trade execution with deterministic governance metadata.
"""

from __future__ import annotations

import hashlib
import json
import math

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any, Mapping, MutableMapping, Sequence
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity, SensoryDriftSnapshot
from src.thinking.adaptation.policy_reflection import (
    PolicyReflectionArtifacts,
    PolicyReflectionBuilder,
)
from src.thinking.adaptation.evolution_manager import EvolutionManager
from src.thinking.adaptation.policy_router import PolicyDecision
from src.trading.gating import DriftSentryDecision, DriftSentryGate, serialise_drift_decision
from src.understanding.belief import BeliefState
from src.understanding.decision_diary import DecisionDiaryEntry, DecisionDiaryStore
from src.understanding.router import BeliefSnapshot, UnderstandingDecision, UnderstandingRouter

__all__ = [
    "ComplianceEvent",
    "ComplianceEventType",
    "ComplianceSeverity",
    "AlphaTradeLoopResult",
    "AlphaTradeLoopOrchestrator",
]


def _coerce_datetime(value: object) -> datetime | None:
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


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        try:
            return _normalise_value(asdict(value))
        except Exception:
            return str(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _normalise_value(value.as_dict())
        except Exception:
            return str(value)
    if isinstance(value, datetime):
        reference = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return reference.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return float(value)
    return value


def _serialise_policy_decision_for_diary(
    decision: PolicyDecision | Mapping[str, Any],
    *,
    recorded_at: datetime,
) -> Mapping[str, Any]:
    if isinstance(decision, Mapping):
        payload = {str(key): value for key, value in decision.items()}
    else:
        payload = {
            "tactic_id": decision.tactic_id,
            "parameters": dict(decision.parameters),
            "selected_weight": float(decision.selected_weight),
            "guardrails": dict(decision.guardrails),
            "rationale": decision.rationale,
            "experiments_applied": list(decision.experiments_applied),
            "reflection_summary": dict(decision.reflection_summary),
            "weight_breakdown": dict(decision.weight_breakdown),
            "fast_weight_metrics": dict(decision.fast_weight_metrics),
            "exploration_metadata": dict(decision.exploration_metadata),
        }
        timestamp = decision.decision_timestamp
        if timestamp is not None:
            payload["decision_timestamp"] = _coerce_datetime(timestamp)
    resolved_timestamp = _coerce_datetime(payload.get("decision_timestamp"))
    if resolved_timestamp is not None:
        reference = resolved_timestamp
    else:
        reference = recorded_at
    payload["decision_timestamp"] = reference.astimezone(timezone.utc).isoformat()
    return payload


def _resolve_recorded_at(
    belief_state: BeliefState | Mapping[str, Any] | None,
    belief_snapshot: BeliefSnapshot,
) -> datetime:
    candidates: tuple[object | None, ...] = (
        getattr(belief_snapshot.regime_state, "timestamp", None),
        getattr(belief_snapshot, "metadata", {}).get("generated_at")
        if isinstance(getattr(belief_snapshot, "metadata", {}), Mapping)
        else None,
        belief_state.generated_at if isinstance(belief_state, BeliefState) else None,
        belief_state.get("generated_at") if isinstance(belief_state, Mapping) else None,
    )
    for candidate in candidates:
        resolved = _coerce_datetime(candidate)
        if resolved is not None:
            return resolved
    return datetime.now(timezone.utc)


def _build_diary_fingerprint(
    *,
    policy_id: str,
    stage: PolicyLedgerStage,
    decision_payload: Mapping[str, Any],
    decision_bundle: UnderstandingDecision,
    belief_snapshot: BeliefSnapshot,
    belief_state: BeliefState | Mapping[str, Any] | None,
    drift_decision: DriftSentryDecision | None,
    metadata: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    notes: Sequence[str] | None,
    recorded_at: datetime,
) -> Mapping[str, Any]:
    decision_payload = decision_bundle.decision
    fingerprint: dict[str, Any] = {
        "policy_id": policy_id,
        "stage": stage.value,
        "recorded_at": recorded_at.astimezone(timezone.utc).isoformat(),
        "decision": _normalise_value(decision_payload),
        "fast_weight_summary": _normalise_value(decision_bundle.fast_weight_summary),
        "fast_weight_metrics": _normalise_value(decision_bundle.fast_weight_metrics),
        "belief_id": belief_snapshot.belief_id,
        "regime_state": _normalise_value(
            {
                "regime": belief_snapshot.regime_state.regime,
                "confidence": belief_snapshot.regime_state.confidence,
                "features": dict(belief_snapshot.regime_state.features),
            }
        ),
        "metadata": _normalise_value(metadata),
        "outcomes": _normalise_value(outcomes),
        "notes": list(notes or ()),
    }
    if isinstance(belief_state, BeliefState):
        fingerprint["belief_state"] = _normalise_value(belief_state)
    elif isinstance(belief_state, Mapping):
        fingerprint["belief_state"] = _normalise_value(belief_state)
    if drift_decision is not None:
        fingerprint["drift_decision"] = _normalise_value(
            serialise_drift_decision(drift_decision, evaluated_at=recorded_at)
        )
    return fingerprint


def _deterministic_entry_id(recorded_at: datetime, fingerprint: Mapping[str, Any]) -> str:
    serialisable = _normalise_value(fingerprint)
    payload = json.dumps(serialisable, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
    token = recorded_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"dd-{token}-{digest}"


class ComplianceEventType(StrEnum):
    """Classification for compliance telemetry emitted by the loop."""

    risk_warning = "risk_warning"
    risk_breach = "risk_breach"
    governance_action = "governance_action"
    governance_promotion = "governance_promotion"


class ComplianceSeverity(StrEnum):
    """Severity grading for compliance events."""

    info = "info"
    warn = "warn"
    critical = "critical"


@dataclass(slots=True, frozen=True)
class ComplianceEvent:
    """Structured record describing a single compliance intervention."""

    event_type: ComplianceEventType
    severity: ComplianceSeverity
    summary: str
    occurred_at: datetime
    policy_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "summary": self.summary,
            "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat(),
        }
        if self.policy_id:
            payload["policy_id"] = self.policy_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class AlphaTradeLoopResult:
    """Composite result emitted after running a full AlphaTrade loop iteration."""

    policy_id: str
    release_stage: PolicyLedgerStage
    decision: PolicyDecision
    decision_bundle: UnderstandingDecision
    drift_decision: DriftSentryDecision
    diary_entry: DecisionDiaryEntry
    reflection: PolicyReflectionArtifacts
    metadata: Mapping[str, Any]
    compliance_events: tuple[ComplianceEvent, ...] = field(default_factory=tuple)


class AlphaTradeLoopOrchestrator:
    """Coordinate policy routing, drift gating, and governance recording."""

    _COUNTERFACTUAL_LIMITS: Mapping[PolicyLedgerStage, Mapping[str, float]] = {
        PolicyLedgerStage.PILOT: {"relative": 0.35},
        PolicyLedgerStage.LIMITED_LIVE: {"relative": 0.20},
    }
    _COUNTERFACTUAL_RELATIVE_KEYS: tuple[str, ...] = (
        "counterfactual_relative_delta_limit",
        "counterfactual_relative_limit",
        "counterfactual_guardrail_relative_limit",
        "counterfactual_max_relative_delta",
        "counterfactual_relative_delta_cap",
        "counterfactual_relative_delta_max",
    )
    _COUNTERFACTUAL_ABSOLUTE_KEYS: tuple[str, ...] = (
        "counterfactual_absolute_delta_limit",
        "counterfactual_absolute_limit",
        "counterfactual_guardrail_absolute_limit",
        "counterfactual_max_absolute_delta",
        "counterfactual_absolute_delta_cap",
        "counterfactual_absolute_delta_max",
    )

    def __init__(
        self,
        *,
        router: UnderstandingRouter,
        diary_store: DecisionDiaryStore,
        drift_gate: DriftSentryGate,
        release_manager: LedgerReleaseManager,
        reflection_builder: PolicyReflectionBuilder | None = None,
        evolution_manager: EvolutionManager | None = None,
        deterministic_mode: bool = False,
    ) -> None:
        self._router = router
        self._diary_store = diary_store
        self._drift_gate = drift_gate
        self._release_manager = release_manager
        self._reflection_builder = reflection_builder or PolicyReflectionBuilder(
            router.policy_router
        )
        self._evolution_manager = evolution_manager
        self._deterministic_mode = deterministic_mode
        # Ensure stage-aware thresholds flow into every DriftSentry evaluation.
        self._drift_gate.attach_threshold_resolver(self._resolve_thresholds)

    def _resolve_thresholds(self, policy_id: str | None) -> Mapping[str, Any] | None:
        thresholds = self._release_manager.resolve_thresholds(policy_id)
        return dict(thresholds) if thresholds is not None else None

    def annotate_diary_entry(
        self, entry_id: str, metadata: Mapping[str, Any] | None
    ) -> DecisionDiaryEntry:
        """Attach additional metadata to a previously recorded diary entry."""

        return self._diary_store.merge_metadata(entry_id, metadata)

    @staticmethod
    def _stage_order(stage: PolicyLedgerStage) -> int:
        if stage is PolicyLedgerStage.EXPERIMENT:
            return 0
        if stage is PolicyLedgerStage.PAPER:
            return 1
        if stage is PolicyLedgerStage.PILOT:
            return 2
        return 3

    @classmethod
    def _more_conservative_stage(
        cls, first: PolicyLedgerStage, second: PolicyLedgerStage
    ) -> PolicyLedgerStage:
        return first if cls._stage_order(first) <= cls._stage_order(second) else second

    def _resolve_stage_sources(
        self, *, resolved_policy_id: str, tactic_id: str
    ) -> Mapping[str, PolicyLedgerStage]:
        policy_stage = self._release_manager.resolve_stage(resolved_policy_id)
        tactic_stage = self._release_manager.resolve_stage(tactic_id)
        effective = self._more_conservative_stage(policy_stage, tactic_stage)
        return {
            "policy": policy_stage,
            "tactic": tactic_stage,
            "effective": effective,
        }

    def _resolve_effective_thresholds(
        self,
        *,
        policy_id: str,
        tactic_id: str,
        effective_stage: PolicyLedgerStage,
    ) -> Mapping[str, Any]:
        policy_thresholds = dict(self._release_manager.resolve_thresholds(policy_id))
        if tactic_id == policy_id:
            policy_thresholds["stage"] = effective_stage.value
            return policy_thresholds

        tactic_thresholds = dict(self._release_manager.resolve_thresholds(tactic_id))

        policy_stage = PolicyLedgerStage.from_value(policy_thresholds.get("stage"))
        tactic_stage = PolicyLedgerStage.from_value(tactic_thresholds.get("stage"))

        if self._stage_order(tactic_stage) < self._stage_order(policy_stage):
            thresholds = tactic_thresholds
        else:
            thresholds = policy_thresholds

        thresholds = dict(thresholds)
        thresholds["stage"] = effective_stage.value
        return thresholds

    @staticmethod
    def _stage_gate_reason(stage: PolicyLedgerStage) -> str | None:
        if stage is PolicyLedgerStage.EXPERIMENT:
            return "release_stage_experiment_requires_paper_or_better"
        if stage is PolicyLedgerStage.PAPER:
            return "release_stage_paper_requires_paper_execution"
        return None

    def _apply_governance_guardrails(
        self,
        decision_bundle: UnderstandingDecision,
        *,
        policy_id: str,
        effective_stage: PolicyLedgerStage,
        stage_sources: Mapping[str, PolicyLedgerStage],
    ) -> UnderstandingDecision:
        decision = decision_bundle.decision
        guardrails = dict(decision.guardrails)

        guardrail_stage_value = guardrails.get("release_stage")
        guardrail_stage = effective_stage
        if guardrail_stage_value is not None:
            try:
                existing_stage = PolicyLedgerStage.from_value(guardrail_stage_value)
            except Exception:
                existing_stage = None
            if existing_stage is not None:
                guardrail_stage = self._more_conservative_stage(
                    effective_stage,
                    existing_stage,
                )

        guardrails["release_stage"] = guardrail_stage.value
        guardrails["governance_policy_stage"] = stage_sources["policy"].value
        guardrails["governance_tactic_stage"] = stage_sources["tactic"].value
        guardrails["governance_release_stage"] = guardrail_stage.value

        stage_reason = self._stage_gate_reason(guardrail_stage)
        if stage_reason:
            guardrails["governance_release_stage_gate"] = stage_reason
        else:
            guardrails.pop("governance_release_stage_gate", None)

        force_required = bool(guardrails.get("force_paper"))
        if stage_reason:
            force_required = True
        guardrails["force_paper"] = force_required

        counterfactual_limits = self._resolve_counterfactual_limits(
            policy_id=policy_id,
            tactic_id=decision.tactic_id,
            effective_stage=guardrail_stage,
        )

        guardrails = self._apply_counterfactual_guardrail(
            guardrails,
            decision,
            stage=guardrail_stage,
            limits=counterfactual_limits,
        )

        if guardrails == decision.guardrails:
            return decision_bundle

        updated_decision = replace(decision, guardrails=guardrails)
        return replace(decision_bundle, decision=updated_decision)

    def _resolve_counterfactual_limits(
        self,
        *,
        policy_id: str | None,
        tactic_id: str,
        effective_stage: PolicyLedgerStage,
    ) -> Mapping[str, float]:
        stage_order = self._stage_order
        candidates: list[tuple[int, Mapping[str, Any]]] = []
        seen_ids: set[str] = set()

        def _add_candidate(identifier: str | None) -> None:
            if not identifier or identifier in seen_ids:
                return
            thresholds = self._release_manager.resolve_thresholds(identifier)
            if not thresholds:
                return
            stage_value = thresholds.get("stage")
            try:
                threshold_stage = PolicyLedgerStage.from_value(stage_value)
            except Exception:
                threshold_stage = effective_stage
            if stage_order(threshold_stage) > stage_order(effective_stage):
                return
            candidates.append((stage_order(threshold_stage), thresholds))
            seen_ids.add(identifier)

        _add_candidate(policy_id)
        if tactic_id != policy_id:
            _add_candidate(tactic_id)

        candidates.sort(key=lambda item: item[0])

        relative_limit: float | None = None
        absolute_limit: float | None = None

        for _, thresholds in candidates:
            if relative_limit is None:
                relative_limit = self._coerce_counterfactual_limit(
                    thresholds,
                    self._COUNTERFACTUAL_RELATIVE_KEYS,
                )
            if absolute_limit is None:
                absolute_limit = self._coerce_counterfactual_limit(
                    thresholds,
                    self._COUNTERFACTUAL_ABSOLUTE_KEYS,
                )
            if relative_limit is not None and absolute_limit is not None:
                break

        defaults = self._COUNTERFACTUAL_LIMITS.get(effective_stage, {})
        if relative_limit is None:
            relative_limit = defaults.get("relative")
        if absolute_limit is None:
            absolute_limit = defaults.get("absolute")

        payload: dict[str, float] = {}
        if relative_limit is not None:
            payload["relative"] = float(relative_limit)
        if absolute_limit is not None:
            payload["absolute"] = float(absolute_limit)
        return payload

    @staticmethod
    def _coerce_counterfactual_limit(
        thresholds: Mapping[str, Any],
        keys: Sequence[str],
    ) -> float | None:
        for key in keys:
            value = thresholds.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric) or numeric < 0.0:
                continue
            return numeric
        return None

    def _apply_counterfactual_guardrail(
        self,
        guardrails: MutableMapping[str, Any],
        decision: PolicyDecision,
        *,
        stage: PolicyLedgerStage,
        limits: Mapping[str, float] | None,
    ) -> MutableMapping[str, Any]:
        breakdown = decision.weight_breakdown
        if not isinstance(breakdown, Mapping) or not breakdown:
            return guardrails

        passive_score = breakdown.get("base_score")
        final_score = breakdown.get("final_score") or decision.selected_weight

        try:
            passive_score_value = float(passive_score)
        except (TypeError, ValueError):
            passive_score_value = None
        try:
            final_score_value = float(final_score)
        except (TypeError, ValueError):
            final_score_value = None

        if passive_score_value is None or final_score_value is None:
            return guardrails

        delta = final_score_value - passive_score_value
        relative_delta: float | None = None
        if passive_score_value != 0.0:
            relative_delta = delta / passive_score_value

        max_relative = None
        max_absolute = None
        if limits:
            max_relative = limits.get("relative")
            max_absolute = limits.get("absolute")
        if max_relative is None or max_absolute is None:
            defaults = self._COUNTERFACTUAL_LIMITS.get(stage, {})
            if max_relative is None:
                max_relative = defaults.get("relative")
            if max_absolute is None:
                max_absolute = defaults.get("absolute")

        breached = False
        if max_relative is not None and relative_delta is not None:
            if abs(relative_delta) > max_relative:
                breached = True
        if max_absolute is not None:
            if abs(delta) > max_absolute:
                breached = True

        guardrail_payload: dict[str, Any] = {
            "stage": stage.value,
            "passive_score": passive_score_value,
            "aggro_score": final_score_value,
            "score_delta": delta,
            "relative_delta": relative_delta,
            "max_relative_delta": max_relative,
            "max_absolute_delta": max_absolute,
            "breached": breached,
        }

        if breached:
            guardrail_payload["reason"] = "counterfactual_guardrail_delta_exceeded"
            guardrail_payload["action"] = "force_paper"
            guardrails["force_paper"] = True

        guardrails["counterfactual_guardrail"] = guardrail_payload
        return guardrails

    def run_iteration(
        self,
        belief_snapshot: BeliefSnapshot,
        *,
        belief_state: BeliefState | Mapping[str, Any] | None = None,
        policy_id: str | None = None,
        outcomes: Mapping[str, Any] | None = None,
        drift_snapshot: SensoryDriftSnapshot | None = None,
        trade: Mapping[str, Any] | None = None,
        notes: Sequence[str] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
        reflection_window: int | None = None,
    ) -> AlphaTradeLoopResult:
        """Execute a full AlphaTrade loop step and persist governance evidence."""

        if drift_snapshot is not None:
            self._drift_gate.update_snapshot(drift_snapshot)

        decision_bundle = self._router.route(belief_snapshot)
        decision = decision_bundle.decision

        resolved_policy_id = policy_id or decision.tactic_id
        stage_sources = self._resolve_stage_sources(
            resolved_policy_id=resolved_policy_id,
            tactic_id=decision.tactic_id,
        )
        stage = stage_sources["effective"]
        decision_bundle = self._apply_governance_guardrails(
            decision_bundle,
            policy_id=resolved_policy_id,
            effective_stage=stage,
            stage_sources=stage_sources,
        )
        decision = decision_bundle.decision

        thresholds = self._resolve_effective_thresholds(
            policy_id=resolved_policy_id,
            tactic_id=decision.tactic_id,
            effective_stage=stage,
        )

        drift_decision = self._evaluate_drift(
            policy_id=resolved_policy_id,
            belief_snapshot=belief_snapshot,
            trade=trade,
            thresholds=thresholds,
        )

        diary_entry = self._record_diary_entry(
            policy_id=resolved_policy_id,
            decision_bundle=decision_bundle,
            belief_snapshot=belief_snapshot,
            belief_state=belief_state,
            outcomes=outcomes,
            trade=trade,
            stage=stage,
            drift_decision=drift_decision,
            thresholds=thresholds,
            notes=notes,
            extra_metadata=extra_metadata,
        )

        adaptation_summary: Mapping[str, Any] | None = None
        if self._evolution_manager is not None:
            adaptation_result = self._evolution_manager.observe_iteration(
                decision=decision,
                stage=stage,
                outcomes=diary_entry.outcomes,
                metadata={
                    "diary_entry_id": diary_entry.entry_id,
                    "policy_id": resolved_policy_id,
                },
                regime_state=belief_snapshot.regime_state,
            )
            if adaptation_result is not None:
                adaptation_summary = adaptation_result.as_dict()
                diary_entry = self._diary_store.merge_metadata(
                    diary_entry.entry_id,
                    {"evolution": adaptation_summary},
                )

        reflection = self._reflection_builder.build(window=reflection_window)

        fast_weight_summary = {
            adapter_id: dict(summary)
            for adapter_id, summary in decision_bundle.fast_weight_summary.items()
        }
        fast_weight_metrics = dict(decision_bundle.fast_weight_metrics)

        guardrail_force_paper = bool(decision.guardrails.get("force_paper"))

        metadata: MutableMapping[str, Any] = {
            "policy_id": resolved_policy_id,
            "release_stage": stage.value,
            "drift": drift_decision.as_dict(),
            "thresholds": dict(thresholds),
            "applied_adapters": decision_bundle.applied_adapters,
            "force_paper": drift_decision.force_paper or guardrail_force_paper,
            "trade_metadata": dict(trade or {}),
            "release_stage_sources": {
                "policy": stage_sources["policy"].value,
                "tactic": stage_sources["tactic"].value,
            },
        }
        metadata.setdefault("guardrails", dict(decision.guardrails))
        fast_weight_enabled = belief_snapshot.fast_weights_enabled
        enabled_value = fast_weight_enabled if isinstance(fast_weight_enabled, bool) else None
        metadata["fast_weight"] = {
            "enabled": enabled_value,
            "metrics": fast_weight_metrics,
            "summary": fast_weight_summary,
            "applied_adapters": list(decision_bundle.applied_adapters),
        }
        if adaptation_summary is not None:
            metadata["evolution"] = adaptation_summary
        if extra_metadata:
            metadata.update({str(key): value for key, value in extra_metadata.items()})

        compliance_events = self._build_compliance_events(
            policy_id=resolved_policy_id,
            stage=stage,
            drift_decision=drift_decision,
        )
        if compliance_events:
            metadata["compliance_events"] = [
                event.as_dict() for event in compliance_events
            ]

        return AlphaTradeLoopResult(
            policy_id=resolved_policy_id,
            release_stage=stage,
            decision=decision,
            decision_bundle=decision_bundle,
            drift_decision=drift_decision,
            diary_entry=diary_entry,
            reflection=reflection,
            metadata=dict(metadata),
            compliance_events=compliance_events,
        )

    def _evaluate_drift(
        self,
        *,
        policy_id: str,
        belief_snapshot: BeliefSnapshot,
        trade: Mapping[str, Any] | None,
        thresholds: Mapping[str, Any],
    ) -> DriftSentryDecision:
        symbol = self._resolve_symbol(trade, belief_snapshot)
        quantity = self._to_float(trade.get("quantity")) if trade else None
        notional = self._to_float(trade.get("notional")) if trade else None
        metadata = dict(trade or {})
        metadata.setdefault("policy_id", policy_id)
        metadata.setdefault("evaluated_at", datetime.now(tz=timezone.utc).isoformat())
        return self._drift_gate.evaluate_trade(
            symbol=symbol,
            strategy_id=policy_id,
            confidence=belief_snapshot.regime_state.confidence,
            quantity=quantity,
            notional=notional,
            metadata=metadata,
            threshold_overrides=thresholds,
        )

    @staticmethod
    def _resolve_symbol(
        trade: Mapping[str, Any] | None,
        belief_snapshot: BeliefSnapshot,
    ) -> str | None:
        if trade is not None:
            candidate = trade.get("symbol")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        meta_symbol = belief_snapshot.metadata.get("symbol")
        if isinstance(meta_symbol, str) and meta_symbol.strip():
            return meta_symbol.strip()
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compliance_severity_from_drift(
        severity: DriftSeverity, *, escalate: bool = False
    ) -> ComplianceSeverity:
        if escalate or severity is DriftSeverity.alert:
            return ComplianceSeverity.critical
        if severity is DriftSeverity.warn:
            return ComplianceSeverity.warn
        return ComplianceSeverity.info

    def _build_compliance_events(
        self,
        *,
        policy_id: str,
        stage: PolicyLedgerStage,
        drift_decision: DriftSentryDecision,
    ) -> tuple[ComplianceEvent, ...]:
        events: list[ComplianceEvent] = []
        metadata: MutableMapping[str, Any] = {
            "release_stage": stage.value,
            "drift_decision": drift_decision.as_dict(),
        }
        requirements = drift_decision.requirements
        stage_gate_reason: str | None = None
        stage_gate_value: str | None = None
        if isinstance(requirements, Mapping):
            candidate_reason = requirements.get("release_stage_gate")
            if isinstance(candidate_reason, str) and candidate_reason.strip():
                stage_gate_reason = candidate_reason.strip()
            candidate_stage = requirements.get("release_stage")
            if isinstance(candidate_stage, str) and candidate_stage.strip():
                stage_gate_value = candidate_stage.strip()
        blocked = tuple(drift_decision.blocked_dimensions)
        if blocked:
            metadata["blocked_dimensions"] = blocked

        reason = drift_decision.reason or "unspecified_reason"
        timestamp = drift_decision.evaluated_at

        if not drift_decision.allowed:
            summary = (
                f"Policy {policy_id} trade blocked at stage {stage.value} "
                f"(severity {drift_decision.severity.value})"
            )
            if blocked:
                summary += f"; blocked dimensions: {', '.join(blocked)}"
            events.append(
                ComplianceEvent(
                    event_type=ComplianceEventType.risk_breach,
                    severity=ComplianceSeverity.critical,
                    summary=summary,
                    occurred_at=timestamp,
                    policy_id=policy_id,
                    metadata={**metadata, "action": "blocked", "reason": reason},
                )
            )
            return tuple(events)

        if drift_decision.force_paper:
            if stage_gate_reason:
                stage_label = stage_gate_value or stage.value
                summary = (
                    f"Policy {policy_id} forced to paper by governance stage {stage_label} "
                    f"(drift severity {drift_decision.severity.value})"
                )
            else:
                summary = (
                    f"Policy {policy_id} forced to paper due to {drift_decision.severity.value}"
                )
            events.append(
                ComplianceEvent(
                    event_type=ComplianceEventType.governance_action,
                    severity=self._compliance_severity_from_drift(
                        drift_decision.severity, escalate=drift_decision.severity is DriftSeverity.alert
                    ),
                    summary=summary,
                    occurred_at=timestamp,
                    policy_id=policy_id,
                    metadata={
                        **metadata,
                        "action": "force_paper",
                        "reason": reason,
                        **(
                            {"release_stage_gate": stage_gate_reason}
                            if stage_gate_reason
                            else {}
                        ),
                    },
                )
            )

        if drift_decision.severity is DriftSeverity.warn:
            summary = f"Policy {policy_id} operating under drift warning"
            events.append(
                ComplianceEvent(
                    event_type=ComplianceEventType.risk_warning,
                    severity=ComplianceSeverity.warn,
                    summary=summary,
                    occurred_at=timestamp,
                    policy_id=policy_id,
                    metadata={**metadata, "action": "warn"},
                )
            )

        return tuple(events)

    def _record_diary_entry(
        self,
        *,
        policy_id: str,
        decision_bundle: UnderstandingDecision,
        belief_snapshot: BeliefSnapshot,
        belief_state: BeliefState | Mapping[str, Any] | None,
        outcomes: Mapping[str, Any] | None,
        trade: Mapping[str, Any] | None,
        stage: PolicyLedgerStage,
        drift_decision: DriftSentryDecision,
        thresholds: Mapping[str, Any],
        notes: Sequence[str] | None,
        extra_metadata: Mapping[str, Any] | None,
    ) -> DecisionDiaryEntry:
        recorded_at = _resolve_recorded_at(belief_state, belief_snapshot)
        regime_timestamp = _coerce_datetime(getattr(belief_snapshot.regime_state, "timestamp", None))
        if regime_timestamp is not None:
            recorded_at = regime_timestamp

        decision_payload = _serialise_policy_decision_for_diary(
            decision_bundle.decision,
            recorded_at=recorded_at,
        )
        decision_timestamp = _coerce_datetime(decision_payload.get("decision_timestamp"))
        entry_timestamp = decision_timestamp or recorded_at
        if self._deterministic_mode:
            try:
                entry_timestamp = self._diary_store._now()  # type: ignore[attr-defined]
            except Exception:
                entry_timestamp = decision_timestamp or recorded_at

        metadata: MutableMapping[str, Any] = {
            "release_stage": stage.value,
            "thresholds": dict(thresholds),
            "applied_adapters": list(decision_bundle.applied_adapters),
            "fast_weight_summary": {
                adapter_id: dict(summary)
                for adapter_id, summary in decision_bundle.fast_weight_summary.items()
            },
            "fast_weight_metrics": dict(decision_bundle.fast_weight_metrics),
        }
        if trade:
            metadata["trade"] = dict(trade)
        if extra_metadata:
            metadata.update({str(key): value for key, value in extra_metadata.items()})

        if drift_decision is not None:
            metadata["drift_decision"] = serialise_drift_decision(
                drift_decision, evaluated_at=entry_timestamp
            )

        outcomes_payload = dict(outcomes or {})

        if self._deterministic_mode:
            metadata = {
                "release_stage": stage.value,
                "thresholds": dict(thresholds),
                "applied_adapters": list(decision_bundle.applied_adapters),
                "fast_weight_summary": {
                    adapter_id: dict(summary)
                    for adapter_id, summary in decision_bundle.fast_weight_summary.items()
                },
                "fast_weight_metrics": dict(decision_bundle.fast_weight_metrics),
            }
            if drift_decision is not None:
                metadata["drift_decision"] = serialise_drift_decision(
                    drift_decision, evaluated_at=entry_timestamp
                )
            outcomes_payload = {}
        notes_tuple: tuple[str, ...] = tuple(str(note).strip() for note in (notes or ()) if str(note).strip())

        fingerprint = _build_diary_fingerprint(
            policy_id=policy_id,
            stage=stage,
            decision_payload=decision_payload,
            decision_bundle=decision_bundle,
            belief_snapshot=belief_snapshot,
            belief_state=belief_state,
            drift_decision=drift_decision,
            metadata=metadata,
            outcomes=outcomes_payload,
            notes=notes_tuple,
            recorded_at=entry_timestamp,
        )
        entry_id = _deterministic_entry_id(entry_timestamp, fingerprint)

        return self._diary_store.record(
            policy_id=policy_id,
            decision=decision_payload,
            regime_state=belief_snapshot.regime_state,
            belief_state=belief_state,
            outcomes=outcomes_payload,
            notes=notes_tuple,
            metadata=metadata,
            entry_id=entry_id,
            recorded_at=entry_timestamp,
        )
