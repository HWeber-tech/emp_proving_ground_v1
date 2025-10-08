"""Legacy portfolio-risk config module removed.

The canonical risk configuration now lives in ``src.config.risk.risk_config``.
Import ``RiskConfig`` from that module instead of relying on this legacy
loader facade.  This stub raises immediately to prevent shim resurrection and
keeps roadmap evidence aligned with the dead-code cleanup plan.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.data_foundation.config.risk_portfolio_config was removed. "
    "Import RiskConfig from src.config.risk.risk_config instead."
)
