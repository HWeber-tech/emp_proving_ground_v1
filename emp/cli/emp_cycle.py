"""Command-line entry point for the lightweight experimentation cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from emp.core import findings_memory, quick_eval, select_next, strategy_factory

from ._emp_cycle_common import (
    DEFAULT_BASELINE,
    DEFAULT_IDEAS_SLICE_DAYS,
    DEFAULT_IDEAS_SYMBOLS,
    ProgressCfg,
    SecondaryConstraint,
    apply_seed,
    atomic_write_json,
    build_data_slice,
    build_progress_cfg,
    detect_git_sha,
    extract_full_metrics,
    fetch_params,
    FuturesTimeout,
    is_progress,
    load_baseline,
    parse_secondary_constraints,
    run_full_backtest,
)

# Backwards-compatible helper aliases retained for downstream imports/tests.
_is_progress = is_progress


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EMP experimentation cycle")
    parser.add_argument("--ideas-json", required=True, help="Path to JSON list of idea parameter dicts")
    parser.add_argument("--quick-threshold", type=float, default=0.5)
    parser.add_argument("--ucb-c", type=float, default=0.2, help="Exploration weight for UCB-lite")
    parser.add_argument("--slice-days", type=int, default=DEFAULT_IDEAS_SLICE_DAYS)
    parser.add_argument(
        "--slice-symbols",
        default=",".join(DEFAULT_IDEAS_SYMBOLS),
        help="Comma separated list of symbols used for the quick slice",
    )
    parser.add_argument("--db-path", default=str(findings_memory.DEFAULT_DB_PATH))
    parser.add_argument(
        "--baseline-json",
        default="data/baseline.json",
        help="Baseline metrics JSON file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip the full test stage")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--kpi", default="sharpe", help="Primary KPI metric key")
    parser.add_argument(
        "--kpi-threshold",
        type=float,
        default=None,
        help="Absolute KPI threshold required for progress (overrides baseline comparison)",
    )
    parser.add_argument(
        "--risk-max-dd",
        type=float,
        default=None,
        help="Maximum allowable absolute drawdown percentage for progress",
    )
    parser.add_argument(
        "--kpi-secondary",
        action="append",
        default=[],
        help="Additional KPI constraint in the form name:op:value (e.g. winrate:>=:0.52)",
    )
    parser.add_argument(
        "--full-timeout-secs",
        type=float,
        default=1200.0,
        help="Timeout for the full backtest evaluation (seconds)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed applied to the run")
    parser.add_argument("--git-sha", default=None, help="Override git SHA metadata (auto-detected if omitted)")
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_ideas(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Ideas JSON must be a list of parameter dictionaries")
    ideas: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Idea at index {idx} is not a dictionary")
        ideas.append(item)
    return ideas


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    apply_seed(args.seed)
    conn = findings_memory.connect(args.db_path)
    ideas = _load_ideas(args.ideas_json)
    data_slice = build_data_slice(args.slice_days, args.slice_symbols)
    secondary_constraints = parse_secondary_constraints(list(args.kpi_secondary))
    cfg = build_progress_cfg(
        str(args.kpi),
        threshold=args.kpi_threshold,
        risk_max_dd=args.risk_max_dd,
        secondary=secondary_constraints,
    )
    metadata_parts: List[str] = []
    if args.seed is not None:
        metadata_parts.append(f"seed:{args.seed}")
    sha = args.git_sha or detect_git_sha()
    if sha:
        metadata_parts.append(f"git:{sha}")
    metadata_note = " ".join(metadata_parts) if metadata_parts else None

    if args.debug:
        print(f"Loaded {len(ideas)} ideas from {args.ideas_json}")

    for params in ideas:
        artefacts = findings_memory.compute_params_artifacts(params)
        novelty = findings_memory.nearest_novelty(conn, params, artefacts=artefacts)
        add_result = findings_memory.add_idea(
            conn,
            params,
            novelty,
            artefacts=artefacts,
            note=metadata_note,
        )
        fid = add_result.id
        if args.debug:
            status = "inserted" if add_result.inserted else "duplicate"
            print(f"Idea {fid} {status} (stage={add_result.stage}) novelty={novelty:.3f}")

        if not add_result.inserted and add_result.stage != "idea":
            continue
        strategy = strategy_factory.make_strategy(params)
        metrics = quick_eval.quick_eval(strategy, data_slice)

        if quick_eval.passes_quick_threshold(metrics, args.quick_threshold):
            findings_memory.update_quick(conn, fid, metrics, metrics["score"])
            print(f"[screened] id={fid} score={metrics['score']:.3f} novelty={novelty:.3f}")
        else:
            findings_memory.append_note(conn, fid, "failed_screen")
            print(f"[filtered] id={fid} score={metrics['score']:.3f}")

    if args.dry_run:
        print("Dry run complete. Skipping selection and full test.")
        return 0

    candidate_id = select_next.pick_next(conn, c=args.ucb_c)
    if candidate_id is None:
        print("No screened candidates available.")
        return 0

    params = fetch_params(conn, candidate_id)
    strategy = strategy_factory.make_strategy(params)
    try:
        result = run_full_backtest(strategy, timeout=args.full_timeout_secs)
        full_metrics = extract_full_metrics(result)
    except FuturesTimeout:
        findings_memory.promote_tested(conn, candidate_id, {"error": "timeout"}, False)
        findings_memory.append_note(conn, candidate_id, "full_eval_error:timeout")
        if metadata_note:
            findings_memory.append_note(conn, candidate_id, metadata_note)
        print(f"[failed] id={candidate_id} reason=timeout")
        return 0
    except Exception as exc:  # pragma: no cover - defensive in tests
        findings_memory.promote_tested(conn, candidate_id, {"error": repr(exc)}, False)
        findings_memory.append_note(conn, candidate_id, f"full_eval_error:{exc!r}")
        if metadata_note:
            findings_memory.append_note(conn, candidate_id, metadata_note)
        print(f"[failed] id={candidate_id} reason=error")
        return 0

    baseline = load_baseline(args.baseline_json)
    progress = is_progress(full_metrics, baseline, cfg)

    findings_memory.promote_tested(conn, candidate_id, full_metrics, progress)
    if metadata_note:
        findings_memory.append_note(conn, candidate_id, metadata_note)

    if progress:
        atomic_write_json(Path(args.baseline_json), full_metrics)
        print(f"[progress] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")
    else:
        print(f"[tested] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
