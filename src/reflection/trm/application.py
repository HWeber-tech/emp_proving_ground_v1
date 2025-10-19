"""Helpers for applying TRM governance decisions to the policy ledger."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStore

logger = logging.getLogger(__name__)

_RIM_AUTO_APPROVAL = "rim-auto"


def apply_auto_applied_suggestions_to_ledger(
    queue_path: Path,
    store: PolicyLedgerStore,
    *,
    release_manager: LedgerReleaseManager,
    auto_approval_tag: str = _RIM_AUTO_APPROVAL,
) -> tuple[str, ...]:
    """Apply auto-approved TRM suggestions to the policy ledger.

    The function scans the governance queue JSONL artifact emitted by the TRM
    runner, identifies suggestions that were automatically approved by the
    governance rule, and records the applied suggestion metadata on the
    corresponding policy ledger record.  Only strategies that already exist in
    the ledger are updated; attempts to create new entries are skipped to avoid
    bypassing manual governance controls.

    Parameters
    ----------
    queue_path:
        Filesystem path to the governance queue JSONL artifact.
    store:
        Policy ledger store backing the runtime release manager.
    release_manager:
        Release manager used to persist updates under the usual governance
        invariants.
    auto_approval_tag:
        Approval label to append when recording TRM-originated changes.

    Returns
    -------
    tuple[str, ...]
        Suggestion identifiers that were applied to the ledger during this
        invocation.
    """

    if not queue_path or not queue_path.exists():
        logger.debug(
            "RIM auto-apply skipped: governance queue missing",
            extra={"rim.auto.queue_path": str(queue_path) if queue_path else None},
        )
        return tuple()

    applied: list[str] = []
    try:
        lines = queue_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - filesystem race
        logger.warning(
            "Unable to read RIM governance queue: %s",
            exc,
            extra={"rim.auto.queue_path": str(queue_path)},
        )
        return tuple()

    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            logger.debug(
                "Skipping malformed governance queue line",
                extra={"rim.auto.queue_path": str(queue_path)},
            )
            continue
        suggestion_id = _extract_suggestion_id(entry)
        if suggestion_id is None:
            continue
        governance = entry.get("governance")
        if not isinstance(governance, Mapping):
            continue
        status = str(governance.get("status", "")).lower()
        auto_apply = governance.get("auto_apply")

        strategy_id = _extract_strategy_id(entry)
        if strategy_id is None:
            continue
        record = store.get(strategy_id)
        if record is None:
            logger.debug(
                "RIM auto-apply suggestion skipped (no ledger entry)",
                extra={
                    "rim.auto.queue_path": str(queue_path),
                    "rim.auto.suggestion_id": suggestion_id,
                    "rim.auto.strategy_id": strategy_id,
                },
            )
            continue

        if status == "auto_applied" and _auto_apply_succeeded(auto_apply):
            existing_entries = _fetch_existing_rim_metadata(record.metadata)
            if suggestion_id in existing_entries:
                continue

            metadata_payload = _build_metadata_payload(entry, governance, auto_apply)
            metadata_update = {
                "rim_auto_apply": existing_entries | {suggestion_id: metadata_payload}
            }

            approvals = list(record.approvals)
            if auto_approval_tag not in approvals:
                approvals.append(auto_approval_tag)

            threshold_overrides: MutableMapping[str, float | str] | None = None
            override_key, override_value = _derive_threshold_override(entry, strategy_id)
            if override_key is not None and override_value is not None:
                overrides = dict(record.threshold_overrides)
                overrides[override_key] = override_value
                threshold_overrides = overrides

            try:
                release_manager.apply_stage_transition(
                    policy_id=strategy_id,
                    tactic_id=record.tactic_id,
                    stage=record.stage,
                    approvals=approvals,
                    evidence_id=record.evidence_id,
                    threshold_overrides=threshold_overrides,
                    metadata=metadata_update,
                    accepted_proposals=(suggestion_id,),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to record RIM auto-apply suggestion: %s",
                    exc,
                    extra={
                        "rim.auto.suggestion_id": suggestion_id,
                        "rim.auto.strategy_id": strategy_id,
                    },
                )
                continue

            applied.append(suggestion_id)
            continue

        rejection_payload = _build_rejection_payload(entry, governance, auto_apply)
        if rejection_payload is None:
            continue

        existing_rejections = _fetch_existing_rejection_metadata(record.metadata)
        if suggestion_id in existing_rejections:
            continue

        rejection_update = {
            "rim_auto_apply_rejections": existing_rejections
            | {suggestion_id: rejection_payload}
        }

        try:
            release_manager.apply_stage_transition(
                policy_id=strategy_id,
                tactic_id=record.tactic_id,
                stage=record.stage,
                approvals=record.approvals,
                evidence_id=record.evidence_id,
                metadata=rejection_update,
                rejected_proposals=(suggestion_id,),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to record RIM suggestion rejection: %s",
                exc,
                extra={
                    "rim.auto.suggestion_id": suggestion_id,
                    "rim.auto.strategy_id": strategy_id,
                },
            )
            continue

    if applied:
        logger.info(
            "Applied %d RIM auto-approved suggestions",
            len(applied),
            extra={
                "rim.auto.queue_path": str(queue_path),
                "rim.auto.applied_ids": applied,
            },
        )
    return tuple(applied)


def _extract_suggestion_id(entry: Mapping[str, object]) -> str | None:
    suggestion_id = entry.get("suggestion_id")
    if not suggestion_id:
        return None
    return str(suggestion_id)


def _extract_strategy_id(entry: Mapping[str, object]) -> str | None:
    payload = entry.get("payload")
    if isinstance(payload, Mapping):
        raw_strategy = payload.get("strategy_id")
        if raw_strategy:
            return str(raw_strategy)
        if str(entry.get("type", "")) == "EXPERIMENT_PROPOSAL":
            candidates = payload.get("strategy_candidates")
            if isinstance(candidates, Sequence) and candidates:
                candidate = candidates[0]
                if candidate:
                    return str(candidate)
    return None


def _auto_apply_succeeded(auto_apply_block: object) -> bool:
    if not isinstance(auto_apply_block, Mapping):
        return False

    applied = auto_apply_block.get("applied")
    if bool(applied) is not True:
        return False

    reasons = auto_apply_block.get("reasons")
    if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
        if any(str(reason).strip() for reason in reasons):
            return False

    evaluation = auto_apply_block.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False

    risk_hits = evaluation.get("risk_hits")
    try:
        risk_hits_value = int(risk_hits)
    except (TypeError, ValueError):
        return False
    if risk_hits_value > 0:
        return False

    budget_remaining = _coerce_finite_float(evaluation.get("budget_remaining"))
    budget_utilisation = _coerce_finite_float(evaluation.get("budget_utilisation"))

    if budget_remaining is None or budget_utilisation is None:
        return False

    if budget_remaining <= 0.0:
        return False

    if not 0.0 <= budget_utilisation <= 1.0:
        return False

    return True


def _coerce_finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fetch_existing_rim_metadata(metadata: Mapping[str, object] | None) -> dict[str, Mapping[str, object]]:
    if not metadata:
        return {}
    rim_metadata = metadata.get("rim_auto_apply")
    if isinstance(rim_metadata, Mapping):
        result: dict[str, Mapping[str, object]] = {}
        for key, value in rim_metadata.items():
            if isinstance(key, str) and isinstance(value, Mapping):
                result[key] = dict(value)
        return result
    return {}


def _fetch_existing_rejection_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, Mapping[str, object]]:
    if not metadata:
        return {}
    rejection_block = metadata.get("rim_auto_apply_rejections")
    if isinstance(rejection_block, Mapping):
        result: dict[str, Mapping[str, object]] = {}
        for key, value in rejection_block.items():
            if isinstance(key, str) and isinstance(value, Mapping):
                result[key] = dict(value)
        return result
    return {}


def _build_metadata_payload(
    entry: Mapping[str, object],
    governance: Mapping[str, object],
    auto_apply: Mapping[str, object],
) -> Mapping[str, object]:
    payload: MutableMapping[str, object] = {
        "type": entry.get("type"),
        "confidence": entry.get("confidence"),
        "payload": entry.get("payload"),
    }
    audit_ids = entry.get("audit_ids")
    if isinstance(audit_ids, Sequence) and audit_ids:
        payload["audit_ids"] = list(audit_ids)
    affected_regimes = entry.get("affected_regimes")
    if isinstance(affected_regimes, Sequence) and affected_regimes:
        payload["affected_regimes"] = [str(regime) for regime in affected_regimes]
    evidence_block = entry.get("evidence")
    if isinstance(evidence_block, Mapping):
        payload["evidence"] = json.loads(json.dumps(evidence_block))
    applied_at = governance.get("applied_at") or governance.get("enqueued_at")
    if applied_at:
        payload["applied_at"] = applied_at
    run_id = governance.get("run_id")
    if run_id:
        payload["run_id"] = run_id
    trace = entry.get("trace")
    if isinstance(trace, Mapping):
        payload["trace"] = json.loads(json.dumps(trace))
    metadata_block = {}
    evaluation = auto_apply.get("evaluation")
    if isinstance(evaluation, Mapping):
        metadata_block["evaluation"] = dict(evaluation)
    reasons = auto_apply.get("reasons")
    if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
        metadata_block["reasons"] = [str(reason) for reason in reasons]
    metadata_block["applied"] = True
    payload["auto_apply"] = metadata_block
    return payload


def _build_rejection_payload(
    entry: Mapping[str, object],
    governance: Mapping[str, object],
    auto_apply: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if not isinstance(auto_apply, Mapping):
        return None
    failure_reasons = ()
    if isinstance(auto_apply, Mapping):
        failure_reasons = _collect_auto_apply_failure_reasons(auto_apply)

    applied_flag = auto_apply.get("applied")
    if isinstance(applied_flag, bool) and applied_flag and not failure_reasons:
        return None

    payload: MutableMapping[str, object] = {
        "type": entry.get("type"),
        "confidence": entry.get("confidence"),
        "payload": entry.get("payload"),
        "status": governance.get("status"),
    }
    audit_ids = entry.get("audit_ids")
    if isinstance(audit_ids, Sequence) and audit_ids:
        payload["audit_ids"] = list(audit_ids)
    affected_regimes = entry.get("affected_regimes")
    if isinstance(affected_regimes, Sequence) and affected_regimes:
        payload["affected_regimes"] = [str(regime) for regime in affected_regimes]
    evidence_block = entry.get("evidence")
    if isinstance(evidence_block, Mapping):
        payload["evidence"] = json.loads(json.dumps(evidence_block))
    run_id = governance.get("run_id")
    if run_id:
        payload["run_id"] = run_id
    evaluated_at = governance.get("enqueued_at")
    if evaluated_at:
        payload["evaluated_at"] = evaluated_at
    trace = entry.get("trace")
    if isinstance(trace, Mapping):
        payload["trace"] = json.loads(json.dumps(trace))

    metadata_block: MutableMapping[str, object] = {"applied": False}
    evaluation = auto_apply.get("evaluation")
    if isinstance(evaluation, Mapping):
        metadata_block["evaluation"] = dict(evaluation)
    if failure_reasons:
        metadata_block["reasons"] = list(failure_reasons)
    else:
        reasons = auto_apply.get("reasons")
        if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
            metadata_block["reasons"] = [str(reason) for reason in reasons]
    payload["auto_apply"] = metadata_block
    return payload


def _derive_threshold_override(
    entry: Mapping[str, object],
    strategy_id: str,
) -> tuple[str | None, float | str | None]:
    suggestion_type = str(entry.get("type", ""))
    if suggestion_type != "WEIGHT_ADJUST":
        return None, None
    payload = entry.get("payload")
    if not isinstance(payload, Mapping):
        return None, None
    delta = payload.get("proposed_weight_delta")
    if isinstance(delta, (int, float)):
        key = f"rim_weight_delta__{strategy_id}"
        try:
            value = float(delta)
        except (TypeError, ValueError):
            return key, None
        return key, value
    return None, None


__all__ = [
    "apply_auto_applied_suggestions_to_ledger",
]
