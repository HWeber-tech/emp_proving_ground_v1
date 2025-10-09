"""Legacy risk manager shim retired in favour of the canonical package.

The roadmap consolidated all authoritative risk facades under ``src.risk`` so
that trading, runtime, and governance modules resolve a single implementation.
This module now raises a clear error to prevent the stale ``src.core`` shim from
masking that migration.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.core.risk.manager was removed. Import RiskManager from src.risk "
    "and use src.config.risk.risk_config.RiskConfig for configuration overrides."
)

