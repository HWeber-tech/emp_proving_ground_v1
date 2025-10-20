"""Queue fill probability estimates for the top of book.

This module fulfils roadmap task **D.2.1** by providing a lightweight helper
that estimates the probability of our resting order at the top of the queue
being filled. The rule of thumb combines our share of the queue with an
exogenous trade-flow factor that captures the expected rate of incoming
liquidity-taking flow.
"""

from __future__ import annotations

import math

__all__ = ["estimate_l1_queue_fill_probability"]


def _coerce_non_negative(value: float | int | None) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(candidate):
        return 0.0
    return max(candidate, 0.0)


def estimate_l1_queue_fill_probability(
    our_size: float | int | None,
    queue_size: float | int | None,
    trade_flow_factor: float | int | None,
    *,
    queue_smoothing: float | int = 1.0,
    clamp: bool = True,
) -> float:
    """Estimate the L1 queue fill probability using the roadmap D.2.1 rule.

    The heuristic assumes that the probability of our order filling at the
    front of the queue is proportional to our share of the resting quantity and
    scales it by an exogenous trade-flow factor that represents the expected
    liquidity-taking pressure.

    ``queue_smoothing`` adds a small positive quantity to the queue size to
    avoid division by zero when the book reports empty queues.
    """

    our = _coerce_non_negative(our_size)
    queue = _coerce_non_negative(queue_size)
    factor = _coerce_non_negative(trade_flow_factor)
    smoothing = _coerce_non_negative(queue_smoothing)

    denominator = queue + smoothing
    if denominator <= 0.0 or factor <= 0.0 or our <= 0.0:
        return 0.0

    probability = factor * (our / denominator)
    if clamp:
        return max(0.0, min(probability, 1.0))
    return probability
