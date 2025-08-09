from __future__ import annotations

"""
Pre-trade risk gates for orders: volume, notional, and exposure caps.

All calculations remain pure and side-effect free; integration points emit
metrics for denials.
"""

from dataclasses import dataclass
from typing import Optional

from src.operational.metrics import inc_pretrade_denial
from src.operational.venue_constraints import align_quantity, align_price


@dataclass
class RiskLimits:
    max_notional: float
    max_volume: int
    max_exposure_per_symbol: int


class PreTradeRiskGate:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        # Simple exposure snapshot; in production use portfolio state
        self._exposure: dict[str, int] = {}

    def check_order(self, symbol: str, side: str, qty: float, price: Optional[float]) -> bool:
        """Return True if allowed, else False and emit denial metric."""
        aq = align_quantity(symbol, qty)
        if aq <= 0:
            inc_pretrade_denial(symbol, "invalid_qty")
            return False
        if aq > self.limits.max_volume:
            inc_pretrade_denial(symbol, "volume_cap")
            return False
        if price is not None:
            ap = align_price(symbol, float(price))
            notional = ap * aq
            if notional > self.limits.max_notional:
                inc_pretrade_denial(symbol, "notional_cap")
                return False
        exposure = self._exposure.get(symbol, 0)
        # Simple symmetric exposure model; BUY increases exposure
        next_exposure = exposure + (aq if side == "1" else -aq)
        if abs(next_exposure) > self.limits.max_exposure_per_symbol:
            inc_pretrade_denial(symbol, "exposure_cap")
            return False
        # Approve and update exposure snapshot
        self._exposure[symbol] = next_exposure
        return True


