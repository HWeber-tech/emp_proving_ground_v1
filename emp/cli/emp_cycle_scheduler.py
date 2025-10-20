"""Automated scheduler for processing experimentation ideas and candidates."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
EVIDENCE_NOTE_QUICK = "evidence:quick-screen"
EVIDENCE_NOTE_FULL = "evidence:full-eval"
EVIDENCE_NOTE_COVERAGE = "coverage:regression"


@dataclass(frozen=True)
class _PlannedCandidate:
    fid: int
    params: Dict[str, Any]
    instrument: str
    quick_score: float
    novelty: float
    ucb_value: float


_INSTRUMENT_KEYS = ("instrument", "symbol", "asset", "ticker", "market")


def _detect_instrument(params: Any) -> str:
    stack: List[Any] = [params]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            for key, value in current.items():
                key_lower = str(key).lower()
                if any(token in key_lower for token in _INSTRUMENT_KEYS):
                    if isinstance(value, str):
                        token = value.strip()
                        if token:
                            return token.upper()
                if isinstance(value, (dict, list, tuple, set)):
                    stack.append(value)
        elif isinstance(current, (list, tuple, set)):
            stack.extend(list(current))
    return "UNKNOWN"


def _fair_share_schedule(
    candidates: List[_PlannedCandidate],
    max_full: Optional[int],
) -> List[_PlannedCandidate]:
    if not candidates:
        return []

    buckets: Dict[str, List[_PlannedCandidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[candidate.instrument].append(candidate)

    for queue in buckets.values():
        queue.sort(
            key=lambda item: (
                -item.ucb_value,
                -item.quick_score,
                -item.novelty,
                item.fid,
            )
        )

    limit = None if max_full is None else int(max_full)
    allocation: List[_PlannedCandidate] = []
    counts: Dict[str, int] = defaultdict(int)

    while buckets and (limit is None or len(allocation) < limit):
        active = [instrument for instrument, queue in buckets.items() if queue]
        if not active:
            break

        best_instrument = min(
            active,
            key=lambda inst: (
                counts[inst],
                -buckets[inst][0].ucb_value,
                -buckets[inst][0].quick_score,
                -buckets[inst][0].novelty,
                buckets[inst][0].fid,
            ),
        )
        candidate = buckets[best_instrument].pop(0)
        allocation.append(candidate)
        counts[best_instrument] += 1
        if not buckets[best_instrument]:
            del buckets[best_instrument]

    return allocation


def _plan_candidate_batch(
    conn,
    *,
    max_full: Optional[int],
    exploration_weight: float,
) -> List[_PlannedCandidate]:
    rows = findings_memory.fetch_candidates(conn)
    candidates: List[_PlannedCandidate] = []
    for fid, params_json, quick_score, novelty in rows:
        try:
            params: Dict[str, Any] = json.loads(params_json)
        except json.JSONDecodeError:
            params = {}
        instrument = _detect_instrument(params)
        quick = float(quick_score or 0.0)
        novel = float(novelty or 0.0)
        ucb_value = select_next.ucb_lite(quick, novel, c=float(exploration_weight))
        candidates.append(
            _PlannedCandidate(
                fid=int(fid),
                params=params,
                instrument=instrument,
                quick_score=quick,
                novelty=novel,
                ucb_value=ucb_value,
            )
        )
    return _fair_share_schedule(candidates, max_full)


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
            _append_note_once(conn, fid, EVIDENCE_NOTE_QUICK)
            _append_note_once(conn, fid, EVIDENCE_NOTE_COVERAGE)
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

    planned = _plan_candidate_batch(
        conn,
        max_full=args.max_full,
        exploration_weight=float(args.ucb_c),
    )

    if not planned:
        print("No screened candidates available for full evaluation.")
        return evaluated, progressed

    if args.dry_run:
        print(f"[dry-run] candidate id={planned[0].fid}")
        return evaluated, progressed

    for candidate in planned:
        candidate_id = candidate.fid
        params = fetch_params(conn, candidate_id)
        strategy = strategy_factory.make_strategy(params)
        try:
            result = run_full_backtest(strategy, timeout=args.full_timeout_secs)
            full_metrics = extract_full_metrics(result)
        except FuturesTimeout:
            print(f"[failed] id={candidate_id} reason=timeout")
            findings_memory.promote_tested(conn, candidate_id, {"error": "timeout"}, False)
            _append_note_once(conn, candidate_id, "full_eval_error:timeout")
            _append_note_once(conn, candidate_id, EVIDENCE_NOTE_FULL)
            _append_note_once(conn, candidate_id, EVIDENCE_NOTE_COVERAGE)
            _append_note_once(conn, candidate_id, metadata_note)
            evaluated += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive in tests
            print(f"[failed] id={candidate_id} reason=error")
            findings_memory.promote_tested(conn, candidate_id, {"error": repr(exc)}, False)
            _append_note_once(conn, candidate_id, f"full_eval_error:{exc!r}")
            _append_note_once(conn, candidate_id, EVIDENCE_NOTE_FULL)
            _append_note_once(conn, candidate_id, EVIDENCE_NOTE_COVERAGE)
            _append_note_once(conn, candidate_id, metadata_note)
            evaluated += 1
            continue

        baseline = load_baseline(args.baseline_json)
        progress = is_progress(full_metrics, baseline, cfg)

        findings_memory.promote_tested(conn, candidate_id, full_metrics, progress)
        _append_note_once(conn, candidate_id, EVIDENCE_NOTE_FULL)
        _append_note_once(conn, candidate_id, EVIDENCE_NOTE_COVERAGE)
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
