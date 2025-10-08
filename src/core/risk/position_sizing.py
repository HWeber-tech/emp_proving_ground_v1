"""Legacy position sizing shim removed in favour of canonical risk helpers."""

from __future__ import annotations

raise ModuleNotFoundError(
    "src.core.risk.position_sizing was removed. Import kelly_fraction and position_size "
    "from src.risk.position_sizing instead."
)
