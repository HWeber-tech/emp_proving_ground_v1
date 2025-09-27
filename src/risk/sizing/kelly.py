"""Kelly sizing helpers with drawdown-aware throttling."""

from __future__ import annotations

from typing import Final

__all__ = ["kelly_fraction", "kelly_position_size"]

_MIN_WIN_RATE: Final[float] = 0.0
_MAX_WIN_RATE: Final[float] = 1.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Return the Kelly fraction for the supplied assumptions.

    The implementation guards against pathological inputs by clamping the win
    rate to ``[0, 1]`` and returning ``0`` when the payoff ratio is non-positive.
    """

    payoff_ratio = float(payoff_ratio)
    if payoff_ratio <= 0:
        return 0.0

    win_rate = _clamp(float(win_rate), _MIN_WIN_RATE, _MAX_WIN_RATE)
    loss_rate = 1.0 - win_rate
    fraction = win_rate - loss_rate / payoff_ratio
    return _clamp(fraction, 0.0, 1.0)


def kelly_position_size(
    equity: float,
    risk_fraction: float,
    stop_loss: float,
    *,
    win_rate: float,
    payoff_ratio: float,
    drawdown_multiplier: float = 1.0,
    min_size: float = 0.0,
    max_size: float | None = None,
) -> float:
    """Calculate a position size using a Kelly fraction with drawdown caps."""

    if stop_loss <= 0:
        raise ValueError("stop_loss must be positive")

    equity = max(0.0, float(equity))
    risk_fraction = max(0.0, float(risk_fraction))
    drawdown_multiplier = max(0.0, float(drawdown_multiplier))

    risk_budget = equity * risk_fraction * drawdown_multiplier
    if risk_budget <= 0:
        return max(min_size, 0.0)

    fraction = kelly_fraction(win_rate, payoff_ratio)
    position = risk_budget / float(stop_loss) * fraction

    position = max(min_size, position)
    if max_size is not None:
        position = min(position, float(max_size))
    return position

