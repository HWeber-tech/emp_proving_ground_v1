"""AlphaTrade loop orchestrator for tactic experimentation and staged promotions.

This module implements the roadmap deliverable that expands the AlphaTrade
Perception → Adaptation → Reflection loop beyond the live-shadow pilot.  It
bridges the understanding router (fast-weight experiments), DriftSentry gating,
policy ledger release stages, and the decision diary so tactic experimentation
feeds paper-trade execution with deterministic governance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity, SensoryDriftSnapshot
from src.thinking.adaptation.policy_reflection import (
    PolicyReflectionArtifacts,
    PolicyReflectionBuilder,
)
from src.thinking.adaptation.evolution_manager import EvolutionManager
from src.thinking.adaptation.policy_router import PolicyDecision
from src.trading.gating import DriftSentryDecision, DriftSentryGate
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

    def __init__(
        self,
        *,
        router: UnderstandingRouter,
        diary_store: DecisionDiaryStore,
        drift_gate: DriftSentryGate,
        release_manager: LedgerReleaseManager,
        reflection_builder: PolicyReflectionBuilder | None = None,
        evolution_manager: EvolutionManager | None = None,
    ) -> None:
        self._router = router
        self._diary_store = diary_store
        self._drift_gate = drift_gate
        self._release_manager = release_manager
        self._reflection_builder = reflection_builder or PolicyReflectionBuilder(
            router.policy_router
        )
        self._evolution_manager = evolution_manager
        # Ensure stage-aware thresholds flow into every DriftSentry evaluation.
        self._drift_gate.attach_threshold_resolver(self._resolve_thresholds)

    def _resolve_thresholds(self, policy_id: str | None) -> Mapping[str, Any] | None:
        thresholds = self._release_manager.resolve_thresholds(policy_id)
        return dict(thresholds) if thresholds is not None else None

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
        stage = self._release_manager.resolve_stage(resolved_policy_id)
        thresholds = self._release_manager.resolve_thresholds(resolved_policy_id)

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

        if self._evolution_manager is not None:
            self._evolution_manager.observe_iteration(
                decision=decision,
                stage=stage,
                outcomes=diary_entry.outcomes,
                metadata={
                    "diary_entry_id": diary_entry.entry_id,
                    "policy_id": resolved_policy_id,
                },
            )

        reflection = self._reflection_builder.build(window=reflection_window)

        metadata: MutableMapping[str, Any] = {
            "policy_id": resolved_policy_id,
            "release_stage": stage.value,
            "drift": drift_decision.as_dict(),
            "thresholds": dict(thresholds),
            "applied_adapters": decision_bundle.applied_adapters,
            "force_paper": drift_decision.force_paper,
            "trade_metadata": dict(trade or {}),
        }
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
                    metadata={**metadata, "action": "force_paper", "reason": reason},
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
        metadata: MutableMapping[str, Any] = {
            "release_stage": stage.value,
            "drift_decision": drift_decision.as_dict(),
            "thresholds": dict(thresholds),
            "applied_adapters": list(decision_bundle.applied_adapters),
            "fast_weight_summary": {
                adapter_id: dict(summary)
                for adapter_id, summary in decision_bundle.fast_weight_summary.items()
            },
        }
        if trade:
            metadata["trade"] = dict(trade)
        if extra_metadata:
            metadata.update({str(key): value for key, value in extra_metadata.items()})

        return self._diary_store.record(
            policy_id=policy_id,
            decision=decision_bundle.decision,
            regime_state=belief_snapshot.regime_state,
            belief_state=belief_state,
            outcomes=dict(outcomes or {}),
            notes=notes,
            metadata=metadata,
        )
