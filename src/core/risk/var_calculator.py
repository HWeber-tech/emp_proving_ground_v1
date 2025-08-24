from __future__ import annotations


def historical_var(returns: list[float], alpha: float = 0.95) -> float:
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    idx = int((1 - alpha) * len(sorted_r))
    idx = max(0, min(len(sorted_r) - 1, idx))
    return abs(sorted_r[idx])
