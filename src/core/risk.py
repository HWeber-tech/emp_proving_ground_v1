"""
Core Risk Manager Port (Protocol)
=================================

Defines a minimal, domain-agnostic interface for risk validation so that
validation and other domain packages do not import src.risk directly.

Concrete implementations should live in higher layers (e.g., src/risk)
and be injected at runtime by orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class RiskConfigDecl:
    """Lightweight risk configuration placeholder for consumers that need it."""

    max_risk_per_trade_pct: Decimal = Decimal("0.02")
    max_leverage: Decimal = Decimal("10.0")
    max_total_exposure_pct: Decimal = Decimal("0.5")
    max_drawdown_pct: Decimal = Decimal("0.25")
    min_position_size: int = 1
    max_position_size: int = 1_000_000


@runtime_checkable
class RiskManagerPort(Protocol):
    """
    Abstract risk manager interface.

    Consumers should pass plain dicts for positions/instruments where possible
    to avoid coupling to concrete types.
    """

    def validate_position(
        self, position: dict[str, object], instrument: dict[str, Any] | Any, equity: Decimal | float
    ) -> bool:
        """
        Validate a position given an instrument and account equity.
        Returns True if the position is acceptable under risk constraints.
        """
        ...


class NoOpRiskManager:
    """Safe default risk manager that approves everything."""

    def __init__(self, config: Optional[RiskConfigDecl] = None) -> None:
        self.config = config or RiskConfigDecl()

    def validate_position(
        self, position: dict[str, object], instrument: dict[str, Any] | Any, equity: Decimal | float
    ) -> bool:
        try:
            # Perform trivial sanity checks without rejecting
            qty = float(position.get("quantity", 0))
            return qty >= 0
        except Exception:
            return True


def is_risk_manager(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, RiskManagerPort)
    except Exception:
        return False
