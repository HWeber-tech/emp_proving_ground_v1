"""
Risk Management Package

Dynamic risk assessment, position sizing, drawdown protection, and risk analytics.
"""

from __future__ import annotations

from typing import Any

# Legacy risk_management facade retained for compatibility.
# Redirects to consolidated `src.core.risk` implementations where applicable.


try:
    from src.core.risk.manager import RiskManager
    from src.core.risk.position_sizing import kelly_fraction as KellyCriterion  # alias
    from src.core.risk.stress_testing import StressTester
    from src.core.risk.var_calculator import VarCalculator as VaRCalculator
except ImportError:  # pragma: no cover
    # Fallbacks if consolidation modules are not present
    RiskManager = object

    def KellyCriterion(*args: Any, **kwargs: Any) -> float:
        return 0.0

    VaRCalculator = object
    StressTester = object


__all__ = ["RiskManager", "KellyCriterion", "VaRCalculator", "StressTester"]
