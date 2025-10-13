"""Shared helpers for EMP experimentation cycle CLIs."""
from __future__ import annotations

import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from emp.core import data_slice

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


def build_data_slice(slice_days: int, symbols_spec: str | Sequence[str] | None) -> Dict[str, Any]:
    symbols: List[str]
    if symbols_spec is None:
        symbols = list(DEFAULT_IDEAS_SYMBOLS)
    elif isinstance(symbols_spec, str):
        tokens = [s.strip() for s in symbols_spec.split(",") if s.strip()]
        symbols = tokens or list(DEFAULT_IDEAS_SYMBOLS)
    else:
        symbols = [str(s).strip() for s in symbols_spec if str(s).strip()]
        if not symbols:
            symbols = list(DEFAULT_IDEAS_SYMBOLS)
    return data_slice.make_slice(symbols, int(slice_days))


def apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy optional
        return
    np.random.seed(seed)  # type: ignore[attr-defined]


def fetch_params(conn, fid: int) -> Dict[str, Any]:
    cursor = conn.execute("SELECT params_json FROM findings WHERE id = ?", (int(fid),))
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Candidate {fid} not found")
    return json.loads(row["params_json"])


def run_full_backtest(strategy: Any, timeout: Optional[float]):
    def _call():
        if hasattr(strategy, "full_backtest") and callable(strategy.full_backtest):
            return strategy.full_backtest()
        return strategy.backtest(None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        return future.result(timeout=timeout)


def extract_full_metrics(result: Any) -> Dict[str, float]:
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


def load_baseline(path: str | os.PathLike[str]) -> Dict[str, float]:
    baseline_path = Path(path)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(baseline_path, DEFAULT_BASELINE)
        return dict(DEFAULT_BASELINE)
    with open(baseline_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {key: float(data.get(key, DEFAULT_BASELINE[key])) for key in DEFAULT_BASELINE}


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def parse_secondary_constraints(raw: Sequence[str]) -> List[SecondaryConstraint]:
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


def build_progress_cfg(
    primary_metric: str,
    *,
    threshold: Optional[float],
    risk_max_dd: Optional[float],
    secondary: Sequence[SecondaryConstraint],
) -> ProgressCfg:
    return ProgressCfg(
        primary_metric=primary_metric,
        threshold=threshold,
        risk_max_dd=risk_max_dd,
        secondary=tuple(secondary),
    )


def compare(op: str, value: float, target: float) -> bool:
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


def is_progress(metrics: Dict[str, float], baseline: Dict[str, float], cfg: ProgressCfg) -> bool:
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
        if not compare(constraint.op, metric_value, constraint.value):
            return False
    return True


def detect_git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
    except Exception:
        return None
    sha = out.decode("utf-8").strip()
    return sha or None


__all__ = [
    "DEFAULT_IDEAS_SLICE_DAYS",
    "DEFAULT_IDEAS_SYMBOLS",
    "DEFAULT_BASELINE",
    "SecondaryConstraint",
    "ProgressCfg",
    "build_data_slice",
    "apply_seed",
    "fetch_params",
    "run_full_backtest",
    "extract_full_metrics",
    "load_baseline",
    "atomic_write_json",
    "parse_secondary_constraints",
    "build_progress_cfg",
    "compare",
    "is_progress",
    "detect_git_sha",
    "FuturesTimeout",
]
