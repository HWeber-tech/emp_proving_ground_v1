from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PortfolioRiskConfig:
    per_asset_cap: float = 1.0  # max absolute exposure per asset (0..1)
    aggregate_cap: float = 2.0  # max sum abs exposures across assets
    usd_beta_cap: float = 1.5  # max absolute USD beta exposure
    var95_cap: float = 0.02  # max 1d VaR allowance (fraction of equity)


_yaml: object | None = None
try:  # pragma: no cover
    _yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    _yaml = None


def load_portfolio_risk_config(path: Optional[str] = None) -> PortfolioRiskConfig:
    if path is None:
        path = os.environ.get("RISK_PORTFOLIO_CONFIG_PATH", "config/risk/portfolio.yaml")
    if _yaml is None or not os.path.exists(path):
        return PortfolioRiskConfig()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = getattr(_yaml, "safe_load")(fh) or {}
        rp = data.get("portfolio_risk", data)
        return PortfolioRiskConfig(
            per_asset_cap=float(rp.get("per_asset_cap", 1.0)),
            aggregate_cap=float(rp.get("aggregate_cap", 2.0)),
            usd_beta_cap=float(rp.get("usd_beta_cap", 1.5)),
            var95_cap=float(rp.get("var95_cap", 0.02)),
        )
    except Exception:
        return PortfolioRiskConfig()
