"""Governance gate that applies replay harness decisions to the policy ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerRecord
from src.thinking.adaptation.replay_harness import StageDecision, TacticEvaluationResult

__all__ = ["AdaptiveGovernanceGate"]


@dataclass(slots=True)
class AdaptiveGovernanceGate:
    """Apply replay evaluation outcomes to the policy ledger with audit metadata."""

    release_manager: LedgerReleaseManager
    evidence_prefix: str = "replay"

    def apply_decision(
        self,
        result: TacticEvaluationResult,
        *,
        evaluation_id: str,
        approvals: Iterable[str] | None = None,
        additional_metadata: Mapping[str, object] | None = None,
        evidence_suffix: str | None = None,
    ) -> PolicyLedgerRecord | None:
        """Apply a single replay decision to the policy ledger."""

        if result.decision is StageDecision.maintain:
            return None

        evidence_id = self._build_evidence_id(result, evaluation_id, evidence_suffix)
        metadata = self._build_metadata(result, evaluation_id, additional_metadata)
        record = self.release_manager.apply_stage_transition(
            policy_id=result.policy_id,
            tactic_id=result.tactic_id,
            stage=result.target_stage,
            approvals=tuple(approvals or ()),
            evidence_id=evidence_id,
            metadata=metadata,
            allow_regression=result.decision is StageDecision.demote,
        )
        return record

    def apply_many(
        self,
        results: Sequence[TacticEvaluationResult],
        *,
        evaluation_id: str,
        approvals: Iterable[str] | None = None,
        additional_metadata: Mapping[str, object] | None = None,
    ) -> tuple[PolicyLedgerRecord, ...]:
        """Apply multiple decisions and return the records that changed."""

        applied: list[PolicyLedgerRecord] = []
        for item in results:
            record = self.apply_decision(
                item,
                evaluation_id=evaluation_id,
                approvals=approvals,
                additional_metadata=additional_metadata,
            )
            if record is not None:
                applied.append(record)
        return tuple(applied)

    def _build_evidence_id(
        self,
        result: TacticEvaluationResult,
        evaluation_id: str,
        suffix: str | None,
    ) -> str:
        parts = [self.evidence_prefix, result.policy_id, evaluation_id]
        if suffix:
            parts.append(suffix)
        return ":".join(parts)

    def _build_metadata(
        self,
        result: TacticEvaluationResult,
        evaluation_id: str,
        additional_metadata: Mapping[str, object] | None,
    ) -> Mapping[str, object]:
        metadata: MutableMapping[str, object] = {
            "evaluation_id": evaluation_id,
            "decision": result.decision.value,
            "current_stage": result.current_stage.value,
            "target_stage": result.target_stage.value,
            "reason": result.reason,
            "snapshot_count": result.snapshot_count,
            "evaluated_at": result.evaluated_at.astimezone(timezone.utc).isoformat(),
            "metrics": dict(result.metrics_summary()),
            "thresholds": dict(result.thresholds_summary()),
        }
        if result.execution_topology is not None:
            metadata["execution_topology"] = result.execution_topology
        if additional_metadata:
            metadata["context"] = dict(additional_metadata)
        metadata.setdefault("recorded_at", datetime.now(tz=timezone.utc).isoformat())
        return metadata
