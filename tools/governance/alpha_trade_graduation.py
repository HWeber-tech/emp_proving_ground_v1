"""CLI for assessing AlphaTrade policy graduation readiness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from src.governance.policy_graduation import (
    PolicyGraduationAssessment,
    PolicyGraduationEvaluator,
)
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerRecord,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.understanding.decision_diary import DecisionDiaryStore
from tools.governance._promotion_helpers import build_log_entry, write_promotion_log


_STAGE_RANK = {
    PolicyLedgerStage.EXPERIMENT: 0,
    PolicyLedgerStage.PAPER: 1,
    PolicyLedgerStage.PILOT: 2,
    PolicyLedgerStage.LIMITED_LIVE: 3,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assess DecisionDiary evidence and the policy ledger to recommend AlphaTrade "
            "stage promotions."
        )
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/governance/policy_ledger.json"),
        help="Path to the policy ledger JSON file (default: artifacts/governance/policy_ledger.json).",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        required=True,
        help="Path to the DecisionDiary JSON store to analyse.",
    )
    parser.add_argument(
        "--policy-id",
        dest="policy_ids",
        action="append",
        default=[],
        help="Specific policy identifier to assess (can be repeated). Defaults to all policies in the ledger or diary.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        help="Limit assessment to decisions recorded within the last N hours.",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation to use for JSON output (default: 2).",
    )
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="Promote ledger stages when recommendations clear all blockers.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("artifacts/governance/policy_promotions.log"),
        help=(
            "Governance promotion log file to append to when --apply is used "
            "(default: artifacts/governance/policy_promotions.log)."
        ),
    )
    return parser


def _resolve_policy_ids(
    explicit: Sequence[str],
    ledger_store: PolicyLedgerStore,
    diary_store: DecisionDiaryStore,
) -> list[str]:
    if explicit:
        return sorted({policy_id.strip() for policy_id in explicit if policy_id.strip()})

    ids = {record.policy_id for record in ledger_store.iter_records()}
    if not ids:
        ids.update(entry.policy_id for entry in diary_store.entries())
    return sorted(ids)


def _render_text(
    assessment,
    *,
    applied_stage: PolicyLedgerStage | None = None,
) -> str:
    metrics = assessment.metrics
    current_metrics = metrics.stage(assessment.current_stage)
    if current_metrics is None and metrics.latest_stage is not None:
        current_metrics = metrics.stage(metrics.latest_stage)

    header = [
        f"Policy {assessment.policy_id}",
        "  current: "
        f"{assessment.current_stage.value}"
        f"{f' (declared {assessment.declared_stage.value})' if assessment.declared_stage else ''}"
        f"{f' (audit {assessment.audit_stage.value})' if assessment.audit_stage else ''}"
        f" -> recommended: {assessment.recommended_stage.value}",
    ]

    lines = header
    lines.append(
        "  decisions: total={total} forced_ratio={forced:.2f} normal={normal} warn={warn} alert={alert}".format(
            total=metrics.total_decisions,
            forced=metrics.forced_ratio,
            normal=metrics.normal,
            warn=metrics.warn,
            alert=metrics.alert,
        )
    )
    if current_metrics is not None:
        lines.append(
            "  stage[{stage}]: forced_ratio={forced:.2f} warn_ratio={warn:.2f} normal_ratio={normal:.2f} streak={streak}".format(
                stage=current_metrics.stage.value,
                forced=current_metrics.forced_ratio,
                warn=current_metrics.warn_ratio,
                normal=current_metrics.normal_ratio,
                streak=metrics.consecutive_normal_latest_stage,
            )
        )
    if assessment.approvals:
        lines.append(f"  approvals: {', '.join(assessment.approvals)}")
    if assessment.audit_gaps:
        lines.append(f"  audit_gaps: {', '.join(assessment.audit_gaps)}")

    for stage in (
        PolicyLedgerStage.PAPER,
        PolicyLedgerStage.PILOT,
        PolicyLedgerStage.LIMITED_LIVE,
    ):
        blockers = assessment.stage_blockers.get(stage, ())
        if blockers:
            joined = ", ".join(blockers)
            lines.append(f"  blockers[{stage.value}]: {joined}")
    if applied_stage is not None:
        lines.append(f"  promotion_applied: {applied_stage.value}")
    return "\n".join(lines)


def _apply_promotions(
    *,
    release_manager: LedgerReleaseManager,
    ledger_store: PolicyLedgerStore,
    assessments: Sequence[PolicyGraduationAssessment],
    log_path: Path | None,
) -> dict[str, PolicyLedgerStage]:
    """Advance ledger stages when recommendations and audit checks allow it."""

    applied: dict[str, PolicyLedgerStage] = {}

    for assessment in assessments:
        policy_id = assessment.policy_id
        record: PolicyLedgerRecord | None = ledger_store.get(policy_id)
        if record is None:
            continue

        recommended = assessment.recommended_stage
        current_stage = record.stage
        if _STAGE_RANK[recommended] <= _STAGE_RANK[current_stage]:
            continue

        blockers = assessment.stage_blockers.get(recommended, ())
        if blockers:
            continue

        record = release_manager.promote(
            policy_id=policy_id,
            tactic_id=record.tactic_id,
            stage=recommended,
            approvals=assessment.approvals or record.approvals,
            evidence_id=assessment.evidence_id or record.evidence_id,
            threshold_overrides=record.threshold_overrides,
            policy_delta=record.policy_delta,
            metadata=record.metadata,
        )
        if log_path is not None:
            posture = release_manager.describe(policy_id)
            log_entry = build_log_entry(record, posture)
            try:
                write_promotion_log(log_path, log_entry)
            except Exception as exc:  # pragma: no cover - filesystem errors are rare
                raise RuntimeError(f"failed to write promotion log: {exc}")
        applied[policy_id] = recommended

    return applied


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ledger_path: Path = args.ledger
    diary_path: Path = args.diary

    if not ledger_path.exists():
        parser.error(f"policy ledger not found at {ledger_path}")
    if not diary_path.exists():
        parser.error(f"decision diary not found at {diary_path}")

    ledger_store = PolicyLedgerStore(ledger_path)
    diary_store = DecisionDiaryStore(diary_path, publish_on_record=False)

    policy_ids = _resolve_policy_ids(args.policy_ids, ledger_store, diary_store)
    if not policy_ids:
        parser.error("no policies available to assess")

    window = timedelta(hours=args.hours) if args.hours is not None else None
    release_manager = LedgerReleaseManager(ledger_store)
    evaluator = PolicyGraduationEvaluator(release_manager, diary_store, window=window)

    assessments = [evaluator.assess(policy_id) for policy_id in policy_ids]
    applied_promotions: dict[str, PolicyLedgerStage] = {}

    log_path: Path | None = args.log_file
    if log_path is not None and str(log_path) == "-":
        log_path = None

    if args.apply:
        try:
            applied_promotions = _apply_promotions(
                release_manager=release_manager,
                ledger_store=ledger_store,
                assessments=assessments,
                log_path=log_path,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.emit_json:
        payload = []
        for assessment in assessments:
            data = assessment.to_dict()
            applied_stage = applied_promotions.get(assessment.policy_id)
            data["applied_stage"] = applied_stage.value if applied_stage else None
            payload.append(data)
        json.dump(payload, sys.stdout, indent=args.indent)
        sys.stdout.write("\n")
        return 0

    for index, assessment in enumerate(assessments):
        if index:
            print()
        applied_stage = applied_promotions.get(assessment.policy_id)
        print(_render_text(assessment, applied_stage=applied_stage))

    if applied_promotions:
        print()
        print("Applied promotions:")
        for policy_id, stage in applied_promotions.items():
            print(f"  - {policy_id}: {stage.value}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
