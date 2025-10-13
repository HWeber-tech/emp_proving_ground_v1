"""Automated scheduler for processing experimentation ideas and candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from emp.core import findings_memory, quick_eval, select_next, strategy_factory

from ._emp_cycle_common import (
    DEFAULT_IDEAS_SLICE_DAYS,
    DEFAULT_IDEAS_SYMBOLS,
    ProgressCfg,
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

DEFAULT_METADATA_NOTE = "scheduler:auto"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process pending experimentation ideas and evaluate screened candidates",
    )
    parser.add_argument("--db-path", default=str(findings_memory.DEFAULT_DB_PATH))
    parser.add_argument(
        "--baseline-json",
        default="data/baseline.json",
        help="Baseline metrics JSON file",
    )
    parser.add_argument("--quick-threshold", type=float, default=0.5)
    parser.add_argument("--ucb-c", type=float, default=0.2, help="Exploration weight for UCB-lite")
    parser.add_argument("--slice-days", type=int, default=DEFAULT_IDEAS_SLICE_DAYS)
    parser.add_argument(
        "--slice-symbols",
        default=",".join(DEFAULT_IDEAS_SYMBOLS),
        help="Comma separated list of symbols used for the quick slice",
    )
    parser.add_argument(
        "--full-timeout-secs",
        type=float,
        default=1200.0,
        help="Timeout for the full backtest evaluation (seconds)",
    )
    parser.add_argument("--max-quick", type=int, default=None, help="Maximum number of idea-stage rows to screen")
    parser.add_argument(
        "--max-full",
        type=int,
        default=1,
        help="Maximum number of screened candidates to evaluate (default: 1)",
    )
    parser.add_argument(
        "--note",
        default=DEFAULT_METADATA_NOTE,
        help="Optional note appended to processed findings (default: scheduler:auto)",
    )
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
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed applied to the run")
    parser.add_argument("--git-sha", default=None, help="Override git SHA metadata (auto-detected if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Run without mutating the findings database")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    return parser.parse_args(list(argv) if argv is not None else None)


def _append_note_once(conn, fid: int, note: str | None) -> None:
    if not note:
        return
    row = conn.execute("SELECT notes FROM findings WHERE id = ?", (int(fid),)).fetchone()
    existing = set()
    if row and row["notes"]:
        existing = {chunk for chunk in str(row["notes"]).split(";") if chunk}
    if note in existing:
        return
    findings_memory.append_note(conn, fid, note)


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.3f}"


def _metadata_note(args: argparse.Namespace) -> Optional[str]:
    tokens: List[str] = []
    if args.note:
        tokens.append(str(args.note))
    if args.seed is not None:
        tokens.append(f"seed:{args.seed}")
    sha = args.git_sha or detect_git_sha()
    if sha:
        tokens.append(f"git:{sha}")
    return " ".join(tokens) if tokens else None


def _screen_pending_ideas(
    conn,
    *,
    data_slice,
    args: argparse.Namespace,
    metadata_note: Optional[str],
) -> int:
    sql = (
        "SELECT id, params_json, novelty, created_at "
        "FROM findings WHERE stage = 'idea' "
        "ORDER BY datetime(created_at) ASC, id ASC"
    )
    params: list[object] = []
    if args.max_quick is not None:
        sql += " LIMIT ?"
        params.append(int(args.max_quick))

    rows = conn.execute(sql, params).fetchall()
    screened = 0

    for row in rows:
        fid = int(row["id"])
        params_payload = json.loads(row["params_json"])
        artefacts = findings_memory.compute_params_artifacts(params_payload)
        novelty = row["novelty"]
        if novelty is None:
            novelty = findings_memory.nearest_novelty(
                conn,
                params_payload,
                artefacts=artefacts,
                exclude_id=fid,
            )
            if not args.dry_run:
                with conn:
                    conn.execute(
                        "UPDATE findings SET novelty = ? WHERE id = ?",
                        (float(novelty), fid),
                    )

        try:
            strategy = strategy_factory.make_strategy(params_payload)
            metrics = quick_eval.quick_eval(strategy, data_slice)
        except Exception as exc:  # pragma: no cover - defensive in tests
            print(f"[screen-error] id={fid} error={exc!r}")
            if not args.dry_run:
                _append_note_once(conn, fid, f"quick_eval_error:{exc!r}")
                _append_note_once(conn, fid, metadata_note)
            continue

        score = metrics.get("score", 0.0)
        novelty_str = _format_value(float(novelty) if novelty is not None else None)

        if quick_eval.passes_quick_threshold(metrics, args.quick_threshold):
            print(f"[screened] id={fid} score={score:.3f} novelty={novelty_str}")
            screened += 1
            if args.dry_run:
                continue
            findings_memory.update_quick(conn, fid, metrics, score)
            _append_note_once(conn, fid, metadata_note)
        else:
            print(f"[filtered] id={fid} score={score:.3f}")
            if args.dry_run:
                continue
            _append_note_once(conn, fid, "failed_screen")
            _append_note_once(conn, fid, metadata_note)

    return screened


def _evaluate_candidates(
    conn,
    *,
    args: argparse.Namespace,
    cfg: ProgressCfg,
    metadata_note: Optional[str],
) -> tuple[int, int]:
    evaluated = 0
    progressed = 0

    if args.max_full is not None and args.max_full <= 0:
        return evaluated, progressed

    while args.max_full is None or evaluated < int(args.max_full):
        candidate_id = select_next.pick_next(conn, c=args.ucb_c)
        if candidate_id is None:
            if evaluated == 0:
                print("No screened candidates available for full evaluation.")
            break

        if args.dry_run:
            print(f"[dry-run] candidate id={candidate_id}")
            break

        params = fetch_params(conn, candidate_id)
        strategy = strategy_factory.make_strategy(params)
        try:
            result = run_full_backtest(strategy, timeout=args.full_timeout_secs)
            full_metrics = extract_full_metrics(result)
        except FuturesTimeout:
            print(f"[failed] id={candidate_id} reason=timeout")
            findings_memory.promote_tested(conn, candidate_id, {"error": "timeout"}, False)
            _append_note_once(conn, candidate_id, "full_eval_error:timeout")
            _append_note_once(conn, candidate_id, metadata_note)
            evaluated += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive in tests
            print(f"[failed] id={candidate_id} reason=error")
            findings_memory.promote_tested(conn, candidate_id, {"error": repr(exc)}, False)
            _append_note_once(conn, candidate_id, f"full_eval_error:{exc!r}")
            _append_note_once(conn, candidate_id, metadata_note)
            evaluated += 1
            continue

        baseline = load_baseline(args.baseline_json)
        progress = is_progress(full_metrics, baseline, cfg)

        findings_memory.promote_tested(conn, candidate_id, full_metrics, progress)
        _append_note_once(conn, candidate_id, metadata_note)

        if progress:
            atomic_write_json(Path(args.baseline_json), full_metrics)
            print(f"[progress] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")
            progressed += 1
        else:
            print(f"[tested] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")

        evaluated += 1

    return evaluated, progressed


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    apply_seed(args.seed)
    conn = findings_memory.connect(args.db_path)
    data_slice = build_data_slice(args.slice_days, args.slice_symbols)
    secondary_constraints = parse_secondary_constraints(list(args.kpi_secondary))
    cfg = build_progress_cfg(
        str(args.kpi),
        threshold=args.kpi_threshold,
        risk_max_dd=args.risk_max_dd,
        secondary=secondary_constraints,
    )
    metadata_note = _metadata_note(args)

    screened = _screen_pending_ideas(conn, data_slice=data_slice, args=args, metadata_note=metadata_note)
    evaluated, progressed = _evaluate_candidates(conn, args=args, cfg=cfg, metadata_note=metadata_note)

    if args.debug:
        print(
            "Summary: screened={screened} evaluated={evaluated} progressed={progressed}".format(
                screened=screened,
                evaluated=evaluated,
                progressed=progressed,
            )
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    sys.exit(main())
