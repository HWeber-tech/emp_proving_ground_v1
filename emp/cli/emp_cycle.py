"""Command-line entry point for the lightweight experimentation cycle."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from emp.core import data_slice, findings_memory, quick_eval, select_next, strategy_factory


DEFAULT_IDEAS_SLICE_DAYS = 60
DEFAULT_IDEAS_SYMBOLS = ["SYN_A", "SYN_B"]
DEFAULT_BASELINE = {
    "sharpe": 0.0,
    "return": 0.0,
    "max_dd": 1e9,
    "winrate": 0.0,
}


@dataclass(frozen=True)
class SecondaryConstraint:
    name: str
    op: str
    value: float


@dataclass(frozen=True)
class ProgressCfg:
    primary_metric: str
    threshold: Optional[float]
    risk_max_dd: Optional[float]
    secondary: tuple[SecondaryConstraint, ...]


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
        symbols = list(DEFAULT_IDEAS_SYMBOLS)
    return data_slice.make_slice(symbols, int(args.slice_days))


def _apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy optional
        return
    np.random.seed(seed)  # type: ignore[attr-defined]


def _fetch_params(conn, fid: int) -> Dict[str, Any]:
    cursor = conn.execute("SELECT params_json FROM findings WHERE id = ?", (int(fid),))
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Candidate {fid} not found")
    return json.loads(row["params_json"])


def _run_full_backtest(strategy: Any, timeout: Optional[float]) -> Any:
    def _call():
        if hasattr(strategy, "full_backtest") and callable(strategy.full_backtest):
            return strategy.full_backtest()
        return strategy.backtest(None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        return future.result(timeout=timeout)


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


def _parse_secondary_constraints(raw: List[str]) -> List[SecondaryConstraint]:
    constraints: List[SecondaryConstraint] = []
    for item in raw:
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid secondary KPI constraint: {item}")
        name, op, value_str = parts
        try:
            value = float(value_str)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid numeric value in constraint '{item}'") from exc
        constraints.append(SecondaryConstraint(name=name, op=op, value=value))
    return constraints


def _progress_cfg(args: argparse.Namespace) -> ProgressCfg:
    return ProgressCfg(
        primary_metric=str(args.kpi),
        threshold=args.kpi_threshold,
        risk_max_dd=args.risk_max_dd,
        secondary=tuple(_parse_secondary_constraints(list(args.kpi_secondary))),
    )


def _compare(op: str, value: float, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    if op == "==":
        return value == target
    if op == "!=":
        return value != target
    raise ValueError(f"Unsupported comparison operator '{op}'")


def _is_progress(metrics: Dict[str, float], baseline: Dict[str, float], cfg: ProgressCfg) -> bool:
    candidate_value = float(metrics.get(cfg.primary_metric, float("-inf")))
    if cfg.threshold is not None:
        target = float(cfg.threshold)
    else:
        target = float(baseline.get(cfg.primary_metric, float("-inf")))
    if candidate_value < target:
        return False

    if cfg.risk_max_dd is not None:
        max_dd = abs(float(metrics.get("max_dd", float("inf"))))
        if max_dd > float(cfg.risk_max_dd):
            return False

    for constraint in cfg.secondary:
        metric_value = float(metrics.get(constraint.name, float("nan")))
        if not _compare(constraint.op, metric_value, constraint.value):
            return False
    return True


def _detect_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    sha = out.decode("utf-8").strip()
    return sha or None


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    _apply_seed(args.seed)
    conn = findings_memory.connect(args.db_path)
    ideas = _load_ideas(args.ideas_json)
    data_slice = _build_data_slice(args)
    cfg = _progress_cfg(args)
    metadata_parts: List[str] = []
    if args.seed is not None:
        metadata_parts.append(f"seed:{args.seed}")
    sha = args.git_sha or _detect_git_sha()
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

    params = _fetch_params(conn, candidate_id)
    strategy = strategy_factory.make_strategy(params)
    try:
        result = _run_full_backtest(strategy, timeout=args.full_timeout_secs)
        full_metrics = _extract_full_metrics(result)
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

    baseline = _load_baseline(args.baseline_json)
    progress = _is_progress(full_metrics, baseline, cfg)

    findings_memory.promote_tested(conn, candidate_id, full_metrics, progress)
    if metadata_note:
        findings_memory.append_note(conn, candidate_id, metadata_note)

    if progress:
        _atomic_write_json(Path(args.baseline_json), full_metrics)
        print(f"[progress] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")
    else:
        print(f"[tested] id={candidate_id} sharpe={full_metrics['sharpe']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
