"""Legacy domain models removed.

Execution-report schemas are now expressed through trading telemetry surfaces.
Import structured payload helpers from ``src.trading.monitoring.portfolio_monitor``
or related trading packages instead of this legacy module.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.domain.models was removed. Use the trading telemetry payloads under "
    "src.trading.monitoring or the canonical risk interfaces instead."
)
