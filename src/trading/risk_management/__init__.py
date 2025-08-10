"""
Risk Management Package

Dynamic risk assessment, position sizing, drawdown protection, and risk analytics.
"""

"""Legacy risk_management facade retained for compatibility.

Redirects to consolidated `src.core.risk` implementations where applicable.
"""

try:
    from src.core.risk.manager import RiskManager  # type: ignore
    from src.core.risk.position_sizing import kelly_fraction as KellyCriterion  # alias
    from src.core.risk.var_calculator import VarCalculator as VaRCalculator  # type: ignore
    from src.core.risk.stress_testing import StressTester  # type: ignore
except Exception:  # pragma: no cover
    # Fallbacks if consolidation modules are not present
    RiskManager = object  # type: ignore
    def KellyCriterion(*args, **kwargs):  # type: ignore
        return 0.0
    class VaRCalculator:  # type: ignore
        pass
    class StressTester:  # type: ignore
        pass

__all__ = [
    'RiskManager',
    'KellyCriterion',
    'VaRCalculator',
    'StressTester'
] 
