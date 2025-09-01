from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Optional, Protocol, cast
from src.sensory.what.volatility_engine import VolConfig

_yaml_module = None
try:
    _yaml_module = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    _yaml_module = None


class _YAMLProtocol(Protocol):
    def safe_load(self, stream: object) -> object: ...


yaml_mod: _YAMLProtocol | None = cast(_YAMLProtocol | None, _yaml_module)
# Canonical volatility surface (Phase 1 canonicalization)



def load_vol_config(path: Optional[str] = None) -> VolConfig:
    """Load vol engine config YAML into VolConfig; fallback to defaults if missing.
    Expected YAML keys mirror VolConfig names with nesting allowed as per the proposal.
    """
    if path is None:
        path = os.environ.get("VOL_CONFIG_PATH", "config/vol/vol_engine.yaml")
    if yaml_mod is None or not os.path.exists(path):
        return VolConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            assert yaml_mod is not None
            raw = yaml_mod.safe_load(fh)
            data = cast(dict[str, object], raw or {})
        ve = _as_map(data.get("vol_engine", data))
        rt = _as_map(ve.get("regime_thresholds", {}))
        sz = _as_map(ve.get("sizing", {}))
        var = _as_map(ve.get("var", {}))
        fb = _as_map(ve.get("fallbacks", {}))
        rg = _as_map(ve.get("regime_gate", {}))
        rc = _as_map(ve.get("risk_controls", {}))
        return VolConfig(
            bar_interval_minutes=_parse_interval(_get_scalar(ve, "bar_interval", "5m")),
            daily_fit_lookback_days=_to_int(_get_scalar(ve, "daily_fit_lookback", "500d"), default=500),
            rv_window_minutes=_parse_interval(_get_scalar(ve, "rv_window", "60m")),
            blend_weight=float(_get_scalar(ve, "blend_weight", 0.7)),
            calm_thr=float(_get_scalar(rt, "calm", 0.08)),
            storm_thr=float(_get_scalar(rt, "storm", 0.18)),
            risk_budget_per_trade=float(_get_scalar(sz, "risk_budget_per_trade", 0.003)),
            k_stop=float(_get_scalar(sz, "k_stop", 1.3)),
            var_confidence=float(_get_scalar(var, "confidence", 0.95)),
            ewma_lambda=float(_get_scalar(fb, "ewma_lambda", 0.94)),
            use_regime_gate=bool(_get_any(rg, "enabled", False)),
            block_regime=str(_get_scalar(rg, "block", "storm")),
            gate_mode=str(_get_scalar(rg, "mode", "block")),
            attenuation_factor=float(_get_scalar(rg, "attenuation_factor", 0.3)),
            brake_scale=float(_get_scalar(rc, "brake_scale", 0.7)),
        )
    except Exception:
        return VolConfig()


def _as_map(v: object) -> dict[str, object]:
    if isinstance(v, Mapping):
        return {str(k): cast(object, val) for k, val in v.items()}
    return {}


def _get_scalar(d: dict[str, object], key: str, default: int | float | str) -> int | float | str:
    v = d.get(key, default)
    if isinstance(v, (int, float, str)):
        return v
    return default


def _get_any(d: dict[str, object], key: str, default: object) -> object:
    return d.get(key, default)


def _parse_interval(s: int | float | str) -> int:
    try:
        s_str = str(s).strip().lower()
        if s_str.endswith("m"):
            return int(s_str[:-1])
        if s_str.endswith("h"):
            return int(s_str[:-1]) * 60
        return int(s_str)
    except Exception:
        return 5


def _to_int(v: int | float | str, default: int) -> int:
    try:
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("d"):
                return int(v[:-1])
        return int(v)
    except Exception:
        return default
