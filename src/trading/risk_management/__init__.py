"""
Risk Management Package

Dynamic risk assessment, position sizing, drawdown protection, and risk analytics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

# Legacy risk_management facade retained for compatibility.
# Redirects to consolidated `src.core.risk` implementations where applicable.


if TYPE_CHECKING:
    from src.risk.manager import RiskManager
    from src.core.risk.position_sizing import kelly_fraction as KellyCriterion  # alias

    class StressTester(Protocol): ...

    class VaRCalculator(Protocol): ...

else:
    try:
        from src.risk.manager import RiskManager
        from src.core.risk.position_sizing import kelly_fraction as KellyCriterion  # alias
    except Exception:  # pragma: no cover

        class RiskManager(Protocol):  # minimal interface placeholder
            ...

        def KellyCriterion(*args: Any, **kwargs: Any) -> float:
            return 0.0

    # Always provide lightweight runtime placeholders for optional tools
    class VaRCalculator(Protocol): ...

    class StressTester(Protocol): ...


__all__ = ["RiskManager", "KellyCriterion", "VaRCalculator", "StressTester"]
