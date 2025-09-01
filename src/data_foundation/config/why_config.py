from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class WhyConfig:
    enable_macro_proximity: bool = True
    enable_yields: bool = True
    # Weights for WHY sub-signals when forming WHY composite
    weight_macro: float = 0.5
    weight_yields: float = 0.5
    # Yield feature toggles
    use_slope_2s10s: bool = True
    use_slope_5s30s: bool = True
    use_curvature_2_10_30: bool = True
    use_parallel_shift: bool = True


_yaml: object | None = None
try:  # pragma: no cover
    _yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    _yaml = None


def load_why_config(path: Optional[str] = None) -> WhyConfig:
    if path is None:
        path = os.environ.get("WHY_CONFIG_PATH", "config/why/why_engine.yaml")
    if _yaml is None or not os.path.exists(path):
        return WhyConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = getattr(_yaml, "safe_load")(fh) or {}
        we = data.get("why_engine", data)
        return WhyConfig(
            enable_macro_proximity=bool(we.get("enable_macro_proximity", True)),
            enable_yields=bool(we.get("enable_yields", True)),
            weight_macro=float(we.get("weights", {}).get("macro", 0.5)),
            weight_yields=float(we.get("weights", {}).get("yields", 0.5)),
            use_slope_2s10s=bool(we.get("yield_features", {}).get("slope_2s10s", True)),
            use_slope_5s30s=bool(we.get("yield_features", {}).get("slope_5s30s", True)),
            use_curvature_2_10_30=bool(we.get("yield_features", {}).get("curvature_2_10_30", True)),
            use_parallel_shift=bool(we.get("yield_features", {}).get("parallel_shift", True)),
        )
    except Exception:
        return WhyConfig()
