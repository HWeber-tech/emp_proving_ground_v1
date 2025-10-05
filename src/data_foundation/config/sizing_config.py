from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Mapping, Optional, cast

# Optional yaml module handle (typed)
_yaml_mod: ModuleType | None
try:
    import yaml as _yaml_runtime

    _yaml_mod = _yaml_runtime
except Exception:  # pragma: no cover
    _yaml_mod = None


@dataclass
class SizingConfig:
    k_exposure: float = 0.8  # base scale for exposure
    sigma_floor: float = 0.05  # below this sigma, no reduction
    sigma_ceiling: float = 0.30  # at/above this sigma, heavy reduction
    regime_multipliers: dict[str, float] | None = (
        None  # e.g., {"calm":1.0,"normal":0.8,"storm":0.5}
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_sizing_config(path: Optional[str] = None) -> SizingConfig:
    if path is None:
        path = os.environ.get("SIZING_CONFIG_PATH", "config/execution/sizing.yaml")
    default_regime: dict[str, float] = {"calm": 1.0, "normal": 0.8, "storm": 0.5}
    if _yaml_mod is None or not os.path.exists(path):
        return SizingConfig(regime_multipliers=default_regime)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = _yaml_mod.safe_load(fh) if _yaml_mod else {}
        data = cast(Mapping[str, object], raw or {})
        sizing_section = _as_mapping(data.get("sizing", data))
        regime_raw = _as_mapping(sizing_section.get("regime_multipliers", default_regime))
        regime_multipliers = {
            str(key): _as_float(value, default_regime.get(str(key), 1.0))
            for key, value in regime_raw.items()
        }
        if not regime_multipliers:
            regime_multipliers = default_regime
        return SizingConfig(
            k_exposure=_as_float(sizing_section.get("k_exposure"), 0.8),
            sigma_floor=_as_float(sizing_section.get("sigma_floor"), 0.05),
            sigma_ceiling=_as_float(sizing_section.get("sigma_ceiling"), 0.30),
            regime_multipliers=regime_multipliers,
        )
    except Exception:
        return SizingConfig(regime_multipliers=default_regime)
