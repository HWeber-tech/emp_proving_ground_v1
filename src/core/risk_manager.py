from __future__ import annotations

from typing import Optional

from src.core.risk.manager import RiskManager as RiskManager  # canonical re-export

_GLOBAL: Optional[RiskManager] = None


def get_global_risk_manager() -> RiskManager:
    """Backward-compatible accessor for a process-wide RiskManager instance."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RiskManager()
    return _GLOBAL


__all__ = ["RiskManager", "get_global_risk_manager"]
