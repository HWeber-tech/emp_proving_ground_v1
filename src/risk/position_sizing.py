"""Canonical position sizing helpers for the risk subsystem."""

from __future__ import annotations

from decimal import Decimal


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute the Kelly Criterion fraction for position sizing."""

    denominator = max(avg_loss, 1e-9)
    edge = max(0.0, min(1.0, win_rate))
    opportunity = avg_win / denominator
    risk = 1.0 - edge
    if opportunity == 0:
        return 0.0
    fraction = (opportunity * edge - risk) / opportunity
    return max(0.0, min(1.0, fraction))


def position_size(balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal) -> Decimal:
    """Determine the allowed position size for a given risk budget."""

    risk_amount = balance * risk_per_trade
    divisor = max(Decimal("1e-9"), stop_loss_pct)
    return risk_amount / divisor


__all__ = ["kelly_fraction", "position_size"]
