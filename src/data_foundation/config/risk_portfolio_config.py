"""Legacy portfolio risk config module removed.

Importing this module raises ``ModuleNotFoundError`` to prevent resurrections of the
old loader.  Use ``src.config.risk.risk_config`` for canonical access.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.data_foundation.config.risk_portfolio_config was removed. "
    "Import RiskConfig from src.config.risk.risk_config instead."
)
