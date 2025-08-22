from __future__ import annotations
from typing import List


def shock_returns(returns: List[float], shock: float = -0.05) -> List[float]:
    return [r + shock for r in returns]


