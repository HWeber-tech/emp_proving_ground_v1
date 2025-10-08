"""CLI for promoting AlphaTrade policies via the policy ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerFeatureFlags,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.understanding.decision_diary import DecisionDiaryStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote PolicyRouter tactics through the AlphaTrade policy ledger "
            "so DriftSentry and release routing inherit audited stage metadata."
        )
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/governance/policy_ledger.json"),
        help="Path to the policy ledger JSON file (default: artifacts/governance/policy_ledger.json).",
    )
    parser.add_argument("--policy-id", required=True, help="Policy identifier to promote.")
    parser.add_argument(
        "--tactic-id",
        help="Optional tactic identifier (defaults to the policy identifier).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help=f"Target promotion stage ({', '.join(stage.value for stage in PolicyLedgerStage)}).",
    )
    parser.add_argument(
        "--approval",
        dest="approvals",
        action="append",
        default=[],
        help="Approval signature to attach to the promotion (repeatable).",
    )
    parser.add_argument(
        "--evidence-id",
        help="DecisionDiary evidence identifier supporting the promotion.",
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        default=[],
        help="Threshold override in key=value form (repeatable).",
    )
    parser.add_argument(
        "--metadata",
        dest="metadata_items",
        action="append",
        default=[],
        help="Additional metadata in key=value form (repeatable).",
    )
    parser.add_argument(
        "--delta-regime",
        help="Optional regime label for the policy delta payload.",
    )
    parser.add_argument(
        "--delta-regime-confidence",
        type=float,
        help="Regime confidence associated with the policy delta.",
    )
    parser.add_argument(
        "--delta-risk-config",
        dest="delta_risk",
        action="append",
        default=[],
        help="RiskConfig override in key=value form for the policy delta (repeatable).",
    )
    parser.add_argument(
        "--delta-guardrail",
        dest="delta_guardrails",
        action="append",
        default=[],
        help="Router guardrail override in key=value form for the policy delta (repeatable).",
    )
    parser.add_argument(
        "--delta-note",
        dest="delta_notes",
        action="append",
        default=[],
        help="Reviewer note to include in the policy delta (repeatable).",
    )
    parser.add_argument(
        "--delta-metadata",
        dest="delta_metadata",
        action="append",
        default=[],
        help="Delta metadata entry in key=value form (repeatable).",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Optional decision diary JSON store for evidence validation.",
    )
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Disable the default requirement for DecisionDiary evidence.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation to use for JSON output (default: 2).",
    )
    return parser


def _parse_mapping_arguments(
    items: Sequence[str],
    *,
    value_factory: Callable[[str], Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in items:
        key, value = _split_key_value(raw)
        result[key] = value_factory(value)
    return result


def _split_key_value(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Expected key=value pair, received {item!r}")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Key component cannot be empty")
    return key, raw_value.strip()


def _coerce_threshold_value(raw: str) -> float | str | None:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _coerce_general_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _build_policy_delta(args: argparse.Namespace) -> PolicyDelta | None:
    risk_overrides = _parse_mapping_arguments(args.delta_risk, value_factory=_coerce_general_value)
    guardrail_overrides = _parse_mapping_arguments(
        args.delta_guardrails,
        value_factory=_coerce_general_value,
    )
    metadata = _parse_mapping_arguments(args.delta_metadata, value_factory=_coerce_general_value)
    notes = tuple(str(note).strip() for note in args.delta_notes if str(note).strip())

    if not any((
        args.delta_regime,
        args.delta_regime_confidence is not None,
        risk_overrides,
        guardrail_overrides,
        notes,
        metadata,
    )):
        return None

    return PolicyDelta(
        regime=args.delta_regime,
        regime_confidence=args.delta_regime_confidence,
        risk_config=risk_overrides,
        router_guardrails=guardrail_overrides,
        notes=notes,
        metadata=metadata,
    )


def _load_diary_store(path: Path | None) -> DecisionDiaryStore | None:
    if path is None:
        return None
    return DecisionDiaryStore(path, publish_on_record=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - argparse already reported the error
        return int(exc.code)

    tactic_id = args.tactic_id or args.policy_id
    try:
        stage = PolicyLedgerStage.from_value(args.stage)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        threshold_overrides = _parse_mapping_arguments(
            args.thresholds,
            value_factory=_coerce_threshold_value,
        )
        metadata = _parse_mapping_arguments(
            args.metadata_items,
            value_factory=_coerce_general_value,
        )
        delta = _build_policy_delta(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        diary_store = _load_diary_store(args.diary)
    except Exception as exc:
        print(f"error: failed to load decision diary: {exc}", file=sys.stderr)
        return 1

    try:
        store = PolicyLedgerStore(args.ledger)
    except Exception as exc:
        print(f"error: failed to load policy ledger: {exc}", file=sys.stderr)
        return 1

    feature_flags = PolicyLedgerFeatureFlags(
        require_diary_evidence=not args.allow_missing_evidence,
    )
    evidence_resolver = diary_store.exists if diary_store else None

    manager = LedgerReleaseManager(
        store,
        feature_flags=feature_flags,
        evidence_resolver=evidence_resolver,
    )

    try:
        record = manager.promote(
            policy_id=args.policy_id,
            tactic_id=tactic_id,
            stage=stage,
            approvals=args.approvals,
            evidence_id=args.evidence_id,
            threshold_overrides=threshold_overrides or None,
            policy_delta=delta,
            metadata=metadata or None,
        )
    except Exception as exc:
        print(f"error: failed to promote policy: {exc}", file=sys.stderr)
        return 1

    posture = manager.describe(record.policy_id)
    payload: dict[str, Any] = {
        "policy_id": record.policy_id,
        "tactic_id": record.tactic_id,
        "stage": record.stage.value,
        "approvals": list(record.approvals),
        "updated_at": record.updated_at.isoformat(),
        "release_posture": posture,
    }
    if record.evidence_id:
        payload["evidence_id"] = record.evidence_id
    if record.threshold_overrides:
        payload["threshold_overrides"] = dict(record.threshold_overrides)
    if record.policy_delta is not None and not record.policy_delta.is_empty():
        payload["policy_delta"] = dict(record.policy_delta.as_dict())
    if record.metadata:
        payload["metadata"] = dict(record.metadata)

    text = json.dumps(payload, indent=args.indent, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
