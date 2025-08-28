from __future__ import annotations

import os
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

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


def load_sizing_config(path: Optional[str] = None) -> SizingConfig:
    if path is None:
        path = os.environ.get("SIZING_CONFIG_PATH", "config/execution/sizing.yaml")
    default_regime = {"calm": 1.0, "normal": 0.8, "storm": 0.5}
    if _yaml_mod is None or not os.path.exists(path):
        return SizingConfig(regime_multipliers=default_regime)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = _yaml_mod.safe_load(fh) or {}
        sz = data.get("sizing", data)
        return SizingConfig(
            k_exposure=float(sz.get("k_exposure", 0.8)),
            sigma_floor=float(sz.get("sigma_floor", 0.05)),
            sigma_ceiling=float(sz.get("sigma_ceiling", 0.30)),
            regime_multipliers=sz.get("regime_multipliers", default_regime) or default_regime,
        )
    except Exception:
        return SizingConfig(regime_multipliers=default_regime)
