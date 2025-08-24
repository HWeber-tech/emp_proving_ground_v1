from __future__ import annotations


def shock_returns(returns: list[float], shock: float = -0.05) -> list[float]:
    return [r + shock for r in returns]
