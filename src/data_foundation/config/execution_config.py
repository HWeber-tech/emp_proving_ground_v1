from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlippageModel:
    base_bps: float = 0.2
    spread_coef: float = 50.0  # bps per 1.0 spread in price units
    imbalance_coef: float = 2.0  # bps per unit imbalance
    sigma_coef: float = 50.0  # bps per 1.0 annualized sigma
    size_coef: float = 5.0  # bps per unit size_ratio


@dataclass
class FeeModel:
    commission_bps: float = 0.1


@dataclass
class ExecutionConfig:
    slippage: SlippageModel = field(default_factory=SlippageModel)
    fees: FeeModel = field(default_factory=FeeModel)


yaml: object | None = None
try:  # pragma: no cover
    # Expose a module-level 'yaml' symbol so tests can monkeypatch ec.yaml
    yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    yaml = None


def load_execution_config(path: Optional[str] = None) -> ExecutionConfig:
    if path is None:
        path = os.environ.get("EXECUTION_CONFIG_PATH", "config/execution/execution.yaml")
    if yaml is None or not os.path.exists(path):
        return ExecutionConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = getattr(yaml, "safe_load")(fh) or {}
        ex = data.get("execution", data)
        sl = ex.get("slippage", {})
        fe = ex.get("fees", {})
        return ExecutionConfig(
            slippage=SlippageModel(
                base_bps=float(sl.get("base_bps", 0.2)),
                spread_coef=float(sl.get("spread_coef", 50.0)),
                imbalance_coef=float(sl.get("imbalance_coef", 2.0)),
                sigma_coef=float(sl.get("sigma_coef", 50.0)),
                size_coef=float(sl.get("size_coef", 5.0)),
            ),
            fees=FeeModel(commission_bps=float(fe.get("commission_bps", 0.1))),
        )
    except Exception:
        return ExecutionConfig()
