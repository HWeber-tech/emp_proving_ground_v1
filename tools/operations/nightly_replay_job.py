#!/usr/bin/env python3
"""Nightly replay harness orchestrator.

This CLI wires the replay evaluation harness into an operations-friendly job that
runs on a schedule, persists governance artefacts, and surfaces sensor drift
summaries. It stitches together the evolution replay evaluator, the policy
ledger, and the decision diary so the roadmap's nightly replay requirement can
ship with deterministic evidence bundles.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evolution.evaluation.datasets import dump_recorded_snapshots, load_recorded_snapshots
from src.governance.adaptive_gate import AdaptiveGovernanceGate
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerFeatureFlags,
    PolicyLedgerRecord,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.artifacts import archive_artifact
from src.sensory.monitoring import evaluate_sensor_drift
from src.thinking.adaptation.policy_router import PolicyDecision, PolicyTactic, RegimeState
from src.thinking.adaptation.replay_harness import StageThresholds, TacticEvaluationResult, TacticReplayHarness
from src.understanding.decision_diary import DecisionDiaryStore

logger = logging.getLogger("tools.operations.nightly_replay_job")

DEFAULT_RUN_ROOT = Path("artifacts/nightly_replay")
DEFAULT_SNAPSHOT_COUNT = 64


@dataclass(frozen=True)
class NightlyReplayContext:
    """Resolved paths and metadata for a replay job invocation."""

    run_dir: Path
    dataset_path: Path
    ledger_path: Path
    diary_path: Path
    drift_report_path: Path
    evaluation_report_path: Path
    timestamp: datetime
    evaluation_id: str


@dataclass(frozen=True)
class RecordedDataset:
    """Container for recorded snapshots and their on-disk location."""

    snapshots: tuple
    path: Path


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory that will contain timestamped nightly replay runs (default: artifacts/nightly_replay).",
    )
    parser.add_argument(
        "--timestamp",
        help="Optional UTC timestamp (YYYYmmddTHHMMSSZ) used for deterministic run directories.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Existing sensory replay dataset (JSONL). When omitted a synthetic dataset is generated for the run.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="Optional override for the policy ledger artefact path.",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Optional override for the decision diary artefact path.",
    )
    parser.add_argument(
        "--drift-report",
        type=Path,
        help="Optional override for the sensor drift summary path.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        help="Optional override for the replay evaluation summary path.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=DEFAULT_SNAPSHOT_COUNT,
        help="Number of synthetic snapshots to generate when --dataset is omitted (default: 64).",
    )
    parser.add_argument(
        "--approvals",
        action="append",
        default=[],
        help="Optional approval identifiers recorded on ledger promotions (can be passed multiple times).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Optional minimum confidence override applied during replay evaluation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level for the orchestrator (default: INFO).",
    )
    return parser.parse_args(argv)


def _resolve_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    cleaned = value.strip()
    if not cleaned:
        return datetime.now(tz=timezone.utc)
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        candidate = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp format: {value}") from exc
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _format_run_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def _ensure_run_context(args: argparse.Namespace) -> NightlyReplayContext:
    timestamp = _resolve_timestamp(args.timestamp)
    run_id = _format_run_id(timestamp)
    run_root = args.run_root.expanduser().resolve()
    run_dir = (run_root / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (args.dataset.expanduser().resolve() if args.dataset else run_dir / "recorded_snapshots.jsonl")
    ledger_path = (args.ledger.expanduser().resolve() if args.ledger else run_dir / "policy_ledger.json")
    diary_path = (args.diary.expanduser().resolve() if args.diary else run_dir / "decision_diary.json")
    drift_report_path = (
        args.drift_report.expanduser().resolve() if args.drift_report else run_dir / "sensor_drift_summary.json"
    )
    evaluation_report_path = (
        args.evaluation_report.expanduser().resolve()
        if args.evaluation_report
        else run_dir / "replay_evaluation.json"
    )

    evaluation_id = f"nightly-replay-{run_id}"

    return NightlyReplayContext(
        run_dir=run_dir,
        dataset_path=dataset_path,
        ledger_path=ledger_path,
        diary_path=diary_path,
        drift_report_path=drift_report_path,
        evaluation_report_path=evaluation_report_path,
        timestamp=timestamp,
        evaluation_id=evaluation_id,
    )


def _generate_synthetic_snapshots(count: int, *, start: datetime) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    base_price = 100.0
    for index in range(max(2, count)):
        ts = start + timedelta(minutes=index)
        angle = index / 4.5
        strength = math.sin(angle) * 0.85 + 0.05 * math.sin(angle * 3.2)
        confidence = 0.78 + 0.18 * math.cos(angle / 2.0)
        confidence = max(0.01, min(0.99, confidence))
        base_price = max(0.5, base_price * (1.0 + 0.0025 * math.sin(angle / 1.5)))
        payloads.append(
            {
                "generated_at": ts.astimezone(timezone.utc).isoformat(),
                "integrated_signal": {
                    "strength": strength,
                    "confidence": confidence,
                },
                "dimensions": {
                    "WHAT": {
                        "metadata": {
                            "last_close": base_price,
                        }
                    },
                    "WHY": {
                        "value": {
                            "signal": strength,
                        }
                    },
                },
                "metadata": {
                    "last_price": base_price,
                    "synthetic": True,
                    "idx": index,
                },
            }
        )
    return payloads


def _load_or_generate_dataset(context: NightlyReplayContext, snapshot_count: int) -> RecordedDataset:
    if context.dataset_path.exists():
        snapshots = load_recorded_snapshots(context.dataset_path)
        if not snapshots:
            logger.warning("Existing dataset at %s is empty; generating synthetic snapshots", context.dataset_path)
        else:
            return RecordedDataset(tuple(snapshots), context.dataset_path)

    logger.info("Generating %s synthetic sensory snapshots", snapshot_count)
    synthetic_payloads = _generate_synthetic_snapshots(
        snapshot_count,
        start=context.timestamp - timedelta(minutes=snapshot_count),
    )
    dump_recorded_snapshots(synthetic_payloads, context.dataset_path)
    snapshots = load_recorded_snapshots(context.dataset_path)
    return RecordedDataset(tuple(snapshots), context.dataset_path)


def _default_tactics() -> tuple[PolicyTactic, ...]:
    return (
        PolicyTactic(
            tactic_id="momentum_long_short",
            base_weight=1.0,
            parameters={
                "entry_threshold": 0.45,
                "exit_threshold": 0.2,
                "risk_fraction": 0.25,
                "min_confidence": 0.55,
            },
            guardrails={"max_notional": 75_000.0},
            regime_bias={"balanced": 1.0, "bull": 1.1},
            description="Momentum baseline tactic for nightly replay",
            tags=("momentum", "nightly-replay"),
        ),
        PolicyTactic(
            tactic_id="mean_reversion_guard",
            base_weight=0.8,
            parameters={
                "entry_threshold": 0.4,
                "exit_threshold": 0.15,
                "risk_fraction": 0.2,
                "min_confidence": 0.5,
            },
            guardrails={"max_notional": 60_000.0},
            regime_bias={"balanced": 1.0, "bear": 1.05},
            description="Mean reversion counterpart tactic",
            tags=("mean-reversion", "nightly-replay"),
        ),
    )


def _build_release_manager(
    ledger_path: Path,
    diary_store: DecisionDiaryStore,
) -> LedgerReleaseManager:
    feature_flags = PolicyLedgerFeatureFlags(require_diary_evidence=True)

    def _evidence_resolver(evidence_id: str) -> bool:
        if not evidence_id:
            return False
        entry_id = evidence_id.split(":")[-1]
        return diary_store.exists(entry_id)

    store = PolicyLedgerStore(ledger_path)
    return LedgerReleaseManager(
        store,
        feature_flags=feature_flags,
        evidence_resolver=_evidence_resolver,
    )


def _record_diary_entry(
    diary_store: DecisionDiaryStore,
    tactic: PolicyTactic,
    result: TacticEvaluationResult,
    *,
    evaluation_id: str,
) -> str:
    metrics = result.metrics_summary()
    thresholds = result.thresholds_summary()
    parameters = dict(tactic.parameters)
    if result.execution_topology and "execution_topology" not in parameters:
        parameters["execution_topology"] = result.execution_topology
    decision = PolicyDecision(
        tactic_id=tactic.tactic_id,
        parameters=parameters,
        selected_weight=tactic.base_weight,
        guardrails=dict(tactic.guardrails),
        rationale=result.reason,
        experiments_applied=(evaluation_id,),
        reflection_summary={
            "decision": result.decision.value,
            "evaluation_id": evaluation_id,
            "metrics": metrics,
            "thresholds": thresholds,
        },
        weight_breakdown={"base_weight": tactic.base_weight},
        fast_weight_metrics={"min_confidence": thresholds.get("min_confidence")},
        decision_timestamp=result.evaluated_at,
    )
    regime_state = RegimeState(
        regime="balanced",
        confidence=0.65,
        features={
            "total_return": metrics.get("total_return", 0.0),
            "volatility": metrics.get("volatility", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
        },
        timestamp=result.evaluated_at,
        volatility=metrics.get("volatility", 0.0),
        volatility_state="normal",
    )
    diary_entry = diary_store.record(
        policy_id=result.policy_id,
        decision=decision,
        regime_state=regime_state,
        outcomes={
            "decision": result.decision.value,
            "target_stage": result.target_stage.value,
            "current_stage": result.current_stage.value,
            "snapshot_count": result.snapshot_count,
        }
        | metrics,
        notes=(f"Nightly replay evaluation {evaluation_id}",),
        metadata={
            "evaluation_id": evaluation_id,
            "tactic": tactic.tactic_id,
            "policy": result.policy_id,
            **(
                {"execution_topology": result.execution_topology}
                if result.execution_topology is not None
                else {}
            ),
        },
    )
    logger.info("Recorded diary entry %s for %s", diary_entry.entry_id, result.policy_id)
    return diary_entry.entry_id


def _apply_decision(
    gate: AdaptiveGovernanceGate,
    result: TacticEvaluationResult,
    *,
    evaluation_id: str,
    approvals: Sequence[str],
    diary_entry_id: str,
) -> PolicyLedgerRecord | None:
    record = gate.apply_decision(
        result,
        evaluation_id=evaluation_id,
        approvals=approvals,
        additional_metadata={"diary_entry": diary_entry_id},
        evidence_suffix=diary_entry_id,
    )
    if record is None:
        logger.info(
            "Ledger decision for %s maintained stage %s",
            result.policy_id,
            result.current_stage.value,
        )
    else:
        logger.info(
            "Ledger updated: %s → %s (decision=%s)",
            result.policy_id,
            record.stage.value,
            result.decision.value,
        )
    return record


def _evaluate_tactics(
    dataset: RecordedDataset,
    release_manager: LedgerReleaseManager,
    *,
    tactics: Iterable[PolicyTactic],
    min_confidence: float | None,
) -> tuple[TacticEvaluationResult, ...]:
    harness = TacticReplayHarness(
        snapshots=dataset.snapshots,
        release_manager=release_manager,
    )
    results: list[TacticEvaluationResult] = []
    for tactic in tactics:
        results.append(
            harness.evaluate_tactic(
                tactic,
                policy_id=tactic.tactic_id,
                min_confidence=min_confidence,
            )
        )
    return tuple(results)


def _build_drift_summary(
    dataset: RecordedDataset,
    path: Path,
) -> Mapping[str, object]:
    if not dataset.snapshots:
        payload = {"generated_at": datetime.now(tz=timezone.utc).isoformat(), "results": []}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    frame = pd.DataFrame(
        {
            "price": [snapshot.price for snapshot in dataset.snapshots],
            "strength": [snapshot.strength for snapshot in dataset.snapshots],
            "confidence": [snapshot.confidence for snapshot in dataset.snapshots],
        }
    )
    total = len(frame)
    baseline_window = max(20, (total * 2) // 3)
    if baseline_window >= total:
        baseline_window = max(10, total - 10)
    evaluation_window = max(5, total - baseline_window)
    if baseline_window + evaluation_window > total:
        baseline_window = max(5, total - evaluation_window)
    summary = evaluate_sensor_drift(
        frame,
        baseline_window=baseline_window,
        evaluation_window=evaluation_window,
        min_observations=max(5, evaluation_window),
        z_threshold=3.0,
    )
    payload = summary.as_dict() | {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "baseline_window": baseline_window,
        "evaluation_window": evaluation_window,
        "total_observations": total,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Recorded sensor drift summary at %s (drift_detected=%s)",
        path,
        summary.exceeded,
    )
    return payload


def _render_stage_thresholds(thresholds: StageThresholds) -> Mapping[str, float]:
    return {
        "promote_total_return": thresholds.promote_total_return,
        "promote_win_rate": thresholds.promote_win_rate,
        "promote_sharpe": thresholds.promote_sharpe,
        "promote_max_drawdown": thresholds.promote_max_drawdown,
        "demote_total_return": thresholds.demote_total_return,
        "demote_win_rate": thresholds.demote_win_rate,
        "demote_sharpe": thresholds.demote_sharpe,
        "demote_max_drawdown": thresholds.demote_max_drawdown,
        "min_trades": float(thresholds.min_trades),
        "min_confidence": thresholds.min_confidence,
    }


def _write_evaluation_report(
    context: NightlyReplayContext,
    dataset: RecordedDataset,
    results: Sequence[TacticEvaluationResult],
    transitions: Sequence[PolicyLedgerRecord],
    drift_summary: Mapping[str, object],
) -> None:
    report: MutableMapping[str, object] = {
        "run_id": context.evaluation_id,
        "generated_at": context.timestamp.astimezone(timezone.utc).isoformat(),
        "dataset": str(dataset.path),
        "results": [],
        "transitions": [],
        "drift_summary": drift_summary,
    }
    for result in results:
        entry: MutableMapping[str, object] = {
            "tactic_id": result.tactic_id,
            "policy_id": result.policy_id,
            "decision": result.decision.value,
            "current_stage": result.current_stage.value,
            "target_stage": result.target_stage.value,
            "snapshot_count": result.snapshot_count,
            "evaluated_at": result.evaluated_at.astimezone(timezone.utc).isoformat(),
            "metrics": result.metrics_summary(),
            "thresholds": _render_stage_thresholds(result.thresholds),
            "reason": result.reason,
        }
        report["results"].append(entry)
    for record in transitions:
        report["transitions"].append(
            {
                "policy_id": record.policy_id,
                "tactic_id": record.tactic_id,
                "stage": record.stage.value,
                "approvals": list(record.approvals),
                "evidence_id": record.evidence_id,
                "updated_at": record.updated_at.astimezone(timezone.utc).isoformat(),
            }
        )
    context.evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    context.evaluation_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote replay evaluation summary to %s", context.evaluation_report_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        context = _ensure_run_context(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    dataset = _load_or_generate_dataset(context, args.snapshot_count)

    diary_store = DecisionDiaryStore(context.diary_path)
    release_manager = _build_release_manager(context.ledger_path, diary_store)
    gate = AdaptiveGovernanceGate(release_manager, evidence_prefix="diary")
    tactics = _default_tactics()

    results = _evaluate_tactics(
        dataset,
        release_manager,
        tactics=tactics,
        min_confidence=args.min_confidence,
    )

    approvals = tuple(str(item).strip() for item in args.approvals if str(item).strip())
    transitions: list[PolicyLedgerRecord] = []
    for tactic, result in zip(tactics, results):
        evidence_id = _record_diary_entry(
            diary_store,
            tactic,
            result,
            evaluation_id=context.evaluation_id,
        )
        record = _apply_decision(
            gate,
            result,
            evaluation_id=context.evaluation_id,
            approvals=approvals,
            diary_entry_id=evidence_id,
        )
        if record is not None:
            transitions.append(record)

    drift_summary = _build_drift_summary(dataset, context.drift_report_path)
    _write_evaluation_report(context, dataset, results, transitions, drift_summary)

    if not context.ledger_path.exists():
        context.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "records": {},
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        context.ledger_path.write_text(json.dumps(placeholder, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Ledger artefact initialised at %s", context.ledger_path)

    archived: dict[str, Path | None] = {
        "diary": archive_artifact(
            "diaries",
            context.diary_path,
            timestamp=context.timestamp,
            run_id=context.evaluation_id,
        ),
        "drift_report": archive_artifact(
            "drift_reports",
            context.drift_report_path,
            timestamp=context.timestamp,
            run_id=context.evaluation_id,
        ),
        "ledger": archive_artifact(
            "ledger_exports",
            context.ledger_path,
            timestamp=context.timestamp,
            run_id=context.evaluation_id,
        ),
    }

    missing_archives = [name for name, target in archived.items() if target is None]
    if missing_archives:
        logger.error(
            "Failed to archive replay artefacts: %s",
            ", ".join(sorted(missing_archives)),
        )
        return 1

    for name, target in archived.items():
        logger.info("Archived %s artefact to %s", name, target)

    logger.info(
        "Nightly replay job completed (run_id=%s, transitions=%d)",
        context.evaluation_id,
        len(transitions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
