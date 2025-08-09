from __future__ import annotations

from typing import Optional
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from src.sensory.dimensions.what.volatility_engine import VolConfig


def load_vol_config(path: Optional[str] = None) -> VolConfig:
    """Load vol engine config YAML into VolConfig; fallback to defaults if missing.
    Expected YAML keys mirror VolConfig names with nesting allowed as per the proposal.
    """
    if path is None:
        path = os.environ.get("VOL_CONFIG_PATH", "config/vol/vol_engine.yaml")
    if yaml is None or not os.path.exists(path):
        return VolConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        ve = data.get("vol_engine", data)
        return VolConfig(
            bar_interval_minutes=_parse_interval(ve.get("bar_interval", "5m")),
            daily_fit_lookback_days=_to_int(ve.get("daily_fit_lookback", "500d"), default=500),
            rv_window_minutes=_parse_interval(ve.get("rv_window", "60m")),
            blend_weight=float(ve.get("blend_weight", 0.7)),
            calm_thr=float(ve.get("regime_thresholds", {}).get("calm", 0.08)),
            storm_thr=float(ve.get("regime_thresholds", {}).get("storm", 0.18)),
            risk_budget_per_trade=float(ve.get("sizing", {}).get("risk_budget_per_trade", 0.003)),
            k_stop=float(ve.get("sizing", {}).get("k_stop", 1.3)),
            var_confidence=float(ve.get("var", {}).get("confidence", 0.95)),
            ewma_lambda=float(ve.get("fallbacks", {}).get("ewma_lambda", 0.94)),
            use_regime_gate=bool(ve.get("regime_gate", {}).get("enabled", False)),
            block_regime=str(ve.get("regime_gate", {}).get("block", "storm")),
        )
    except Exception:
        return VolConfig()


def _parse_interval(s: str) -> int:
    try:
        s = str(s).strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        return int(s)
    except Exception:
        return 5


def _to_int(v, default: int) -> int:
    try:
        if isinstance(v, str) and v.endswith("d"):
            return int(v[:-1])
        return int(v)
    except Exception:
        return default


