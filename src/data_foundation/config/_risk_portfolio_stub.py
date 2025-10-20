"""Compatibility shim providing the legacy portfolio risk loader."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass
class PortfolioRiskConfig:
    per_asset_cap: float = 1.0
    aggregate_cap: float = 2.0
    usd_beta_cap: float = 1.5
    var95_cap: float = 0.02
    sector_exposure_limits: Mapping[str, float] = field(default_factory=dict)


yaml: object | None = None
try:  # pragma: no cover - optional dependency
    yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    yaml = None


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_portfolio_risk_config(path: Optional[str] = None) -> PortfolioRiskConfig:
    target_path = path or os.environ.get(
        "PORTFOLIO_RISK_CONFIG_PATH", "config/risk/portfolio.yaml"
    )
    if yaml is None or not os.path.exists(target_path):
        return PortfolioRiskConfig()

    try:
        with open(target_path, "r", encoding="utf-8") as fh:
            payload = getattr(yaml, "safe_load")(fh) or {}
    except Exception:  # pragma: no cover - best effort fallback
        return PortfolioRiskConfig()

    risk_section = payload.get("portfolio_risk", payload)
    per_asset_cap = _coerce_float(risk_section.get("per_asset_cap"), 1.0)
    aggregate_cap = _coerce_float(risk_section.get("aggregate_cap"), 2.0)
    usd_beta_cap = _coerce_float(risk_section.get("usd_beta_cap"), 1.5)
    var95_cap = _coerce_float(risk_section.get("var95_cap"), 0.02)

    sector_limits_raw = risk_section.get("sector_limits") or {}
    limits: dict[str, float] = {}
    if isinstance(sector_limits_raw, Mapping):
        for raw_sector, raw_limit in sector_limits_raw.items():
            if raw_sector is None:
                continue
            key = str(raw_sector).strip()
            if not key:
                continue
            limits[key.upper()] = _coerce_float(raw_limit, 0.0)

    return PortfolioRiskConfig(
        per_asset_cap=per_asset_cap,
        aggregate_cap=aggregate_cap,
        usd_beta_cap=usd_beta_cap,
        var95_cap=var95_cap,
        sector_exposure_limits=limits,
    )


__all__ = ["PortfolioRiskConfig", "load_portfolio_risk_config"]
