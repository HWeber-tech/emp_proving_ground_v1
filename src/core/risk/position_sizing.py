from __future__ import annotations

from decimal import Decimal


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    b = avg_win / max(avg_loss, 1e-9)
    p = max(0.0, min(1.0, win_rate))
    q = 1 - p
    frac = (b * p - q) / b if b != 0 else 0.0
    return max(0.0, min(1.0, frac))


def position_size(balance: Decimal, risk_per_trade: Decimal, stop_loss_pct: Decimal) -> Decimal:
    risk_amount = balance * risk_per_trade
    denom = max(Decimal("1e-9"), stop_loss_pct)
    return risk_amount / denom


