"""Command-line entry point for the lightweight experimentation cycle."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from emp.core import findings_memory, quick_eval, select_next, strategy_factory


DEFAULT_IDEAS_SLICE_DAYS = 60
DEFAULT_IDEAS_SYMBOLS = ["SYN_A", "SYN_B"]
DEFAULT_BASELINE = {
    "sharpe": 0.0,
    "return": 0.0,
    "max_dd": 1e9,
    "winrate": 0.0,
}


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
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_ideas(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
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


def _build_data_slice(args: argparse.Namespace) -> Dict[str, Any]:
    symbols = [s.strip() for s in str(args.slice_symbols).split(",") if s.strip()]
    if not symbols:
        symbols = DEFAULT_IDEAS_SYMBOLS
    return {"days": int(args.slice_days), "symbols": symbols}


def _append_note(conn, fid: int, note: str) -> None:
    with conn:
        conn.execute(
            """
            UPDATE findings
               SET notes = CASE WHEN notes IS NULL OR notes = ''
                                 THEN ?
                                 ELSE notes || ';' || ? END
             WHERE id = ?
            """,
            (note, note, int(fid)),
        )


def _fetch_params(conn, fid: int) -> Dict[str, Any]:
    cursor = conn.execute("SELECT params_json FROM findings WHERE id = ?", (int(fid),))
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Candidate {fid} not found")
    return json.loads(row["params_json"])


def _run_full_backtest(strategy: Any) -> Any:
    if hasattr(strategy, "full_backtest") and callable(strategy.full_backtest):
        return strategy.full_backtest()
    return strategy.backtest(None)


def _extract_full_metrics(result: Any) -> Dict[str, float]:
    sharpe = float(getattr(result, "sharpe", 0.0))
    total_return = float(
        getattr(result, "total_return", getattr(result, "return_pct", getattr(result, "return", 0.0)))
    )
    max_dd = float(getattr(result, "max_drawdown", getattr(result, "max_dd", 0.0)))
    winrate = float(getattr(result, "winrate", 0.0))
    return {
        "sharpe": sharpe,
        "return": total_return,
        "max_dd": max_dd,
        "winrate": winrate,
    }


def _load_baseline(path: str | os.PathLike[str]) -> Dict[str, float]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(baseline_path, DEFAULT_BASELINE)
        return dict(DEFAULT_BASELINE)
    with open(baseline_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {key: float(data.get(key, DEFAULT_BASELINE[key])) for key in DEFAULT_BASELINE}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _is_progress(metrics: Dict[str, float], baseline: Dict[str, float]) -> bool:
    if metrics.get("sharpe", float("-inf")) < baseline.get("sharpe", float("-inf")):
        return False
    if abs(metrics.get("max_dd", 0.0)) > abs(baseline.get("max_dd", float("inf"))):
        return False
    return True


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    conn = findings_memory.connect(args.db_path)
    ideas = _load_ideas(args.ideas_json)
    data_slice = _build_data_slice(args)

    if args.debug:
        print(f"Loaded {len(ideas)} ideas from {args.ideas_json}")

    for params in ideas:
        novelty = findings_memory.nearest_novelty(conn, params)
        fid = findings_memory.add_idea(conn, params, novelty)
        if args.debug:
            print(f"Idea {fid} inserted with novelty {novelty:.3f}")

        strategy = strategy_factory.make_strategy(params)
        metrics = quick_eval.quick_eval(strategy, data_slice)

        if quick_eval.passes_quick_threshold(metrics, args.quick_threshold):
            findings_memory.update_quick(conn, fid, metrics, metrics["score"])
            print(f"[screened] id={fid} score={metrics['score']:.3f} novelty={novelty:.3f}")
        else:
            _append_note(conn, fid, "failed_screen")
            print(f"[filtered] id={fid} score={metrics['score']:.3f}")

    if args.dry_run:
        print("Dry run complete. Skipping selection and full test.")
        return 0

    candidate_id = select_next.pick_next(conn, c=args.ucb_c)
    if candidate_id is None:
        print("No screened candidates available.")
        return 0

    params = _fetch_params(conn, candidate_id)
    strategy = strategy_factory.make_strategy(params)
    result = _run_full_backtest(strategy)
    full_metrics = _extract_full_metrics(result)

    baseline = _load_baseline(args.baseline_json)
    progress = _is_progress(full_metrics, baseline)

    findings_memory.promote_tested(conn, candidate_id, full_metrics, progress)

    if progress:
        _atomic_write_json(Path(args.baseline_json), full_metrics)
        print(f"[progress] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")
    else:
        print(f"[tested] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
