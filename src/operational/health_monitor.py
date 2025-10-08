"""Legacy operational health monitor removed.

Operational readiness and health telemetry now live under ``src.operations``
modules (``event_bus_health``, ``cache_health``, ``data_backbone`` and friends).
Import from those surfaces instead of the deprecated operational shim.
"""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.operational.health_monitor was removed. Use the canonical operations "
    "health modules under src.operations.* instead."
)
