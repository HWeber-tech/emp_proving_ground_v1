"""Quick evaluation helpers for candidate strategies."""
from __future__ import annotations

from typing import Any, Dict


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def quick_eval(strategy: Any, data_slice: Any) -> Dict[str, float]:
    """Run a lightweight backtest slice and compute quick metrics."""

    result = strategy.backtest(data_slice)
    gross_profit = float(getattr(result, "gross_profit", 0.0))
    gross_loss = float(getattr(result, "gross_loss", 0.0))
    max_drawdown = float(getattr(result, "max_drawdown", 0.0))
    sharpe = float(getattr(result, "sharpe", 0.0))

    loss_magnitude = abs(gross_loss)
    profit_factor = _safe_divide(gross_profit, loss_magnitude)
    profit_factor = max(0.0, profit_factor)

    capped_pf = min(profit_factor, 3.0)
    capped_sharpe = max(-3.0, min(3.0, sharpe))
    max_drawdown_abs = abs(max_drawdown)

    score = 0.6 * capped_pf + 0.3 * capped_sharpe - 0.1 * (max_drawdown_abs / 100.0)

    return {
        "profit_factor": profit_factor,
        "max_drawdown_abs": max_drawdown_abs,
        "sharpe": sharpe,
        "score": score,
    }


def passes_quick_threshold(metrics: Dict[str, float], threshold: float = 0.5) -> bool:
    """Return True if the quick score clears the provided threshold."""

    return metrics.get("score", float("-inf")) >= threshold


__all__ = ["quick_eval", "passes_quick_threshold"]
