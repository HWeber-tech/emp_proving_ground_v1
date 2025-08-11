from __future__ import annotations

from typing import Any, Dict, Optional


class RiskManager:
    """Consolidated risk management baseline.

    This minimal implementation satisfies imports and provides simple hooks that
    higher-level components can call without failing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def assess_risk(self, position_size: float, symbol: str) -> Dict[str, Any]:
        return {"risk_level": "low", "max_position": position_size, "symbol": symbol}

    def validate_order(self, order: Dict[str, Any]) -> bool:
        return True


_GLOBAL: Optional[RiskManager] = None


def get_global_risk_manager() -> RiskManager:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = RiskManager()
    return _GLOBAL


