"""Planner edge realisation gate for Phase F requirements.

This module implements the roadmap gate **F.1.3**, ensuring the MuZero-lite
planner's imagined edge correlates with the realised trading edge on a hold-out
session.  The gate evaluates the Pearson correlation between imagined and
realised edge series (optionally weighted) and reports whether the result clears
an operator-supplied threshold (defaulting to 0.20).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math

__all__ = ["PlannerEdgeGateDecision", "evaluate_planner_edge_gate"]


@dataclass(frozen=True, slots=True)
class PlannerEdgeGateDecision:
    """Outcome of the planner edge realisation gate."""

    passed: bool
    correlation: float | None
    minimum_correlation: float
    valid_pairs: int
    total_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "correlation": self.correlation,
            "minimum_correlation": self.minimum_correlation,
            "valid_pairs": self.valid_pairs,
            "total_pairs": self.total_pairs,
        }


def _coerce_float(value: float | int | None, *, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("edge values must be numeric or None") from exc
    if not math.isfinite(numeric):
        return None if allow_none else float("nan")
    return numeric


def _validate_threshold(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("minimum_correlation must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError("minimum_correlation must be finite")
    return numeric


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        total += value * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return total / weight_sum


def _weighted_correlation(pairs: Sequence[tuple[float, float, float]]) -> float | None:
    if len(pairs) < 2:
        return None

    weights = [weight for _, _, weight in pairs]
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]

    mean_x = _weighted_mean(xs, weights)
    mean_y = _weighted_mean(ys, weights)

    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    weight_sum = 0.0

    for x, y, weight in pairs:
        centred_x = x - mean_x
        centred_y = y - mean_y
        cov += weight * centred_x * centred_y
        var_x += weight * centred_x * centred_x
        var_y += weight * centred_y * centred_y
        weight_sum += weight

    if weight_sum <= 0.0:
        return None

    var_x /= weight_sum
    var_y /= weight_sum

    if var_x <= 0.0 or var_y <= 0.0:
        return None

    cov /= weight_sum
    denominator = math.sqrt(var_x * var_y)
    if denominator <= 0.0:
        return None

    correlation = cov / denominator
    if correlation > 1.0:
        return 1.0
    if correlation < -1.0:
        return -1.0
    return correlation


def evaluate_planner_edge_gate(
    imagined_edge_bps: Sequence[float | int | None],
    realised_edge_bps: Sequence[float | int | None],
    *,
    minimum_correlation: float = 0.20,
    weights: Sequence[float | int | None] | None = None,
) -> PlannerEdgeGateDecision:
    """Evaluate the planner edge realisation correlation gate.

    Parameters
    ----------
    imagined_edge_bps:
        Sequence of imagined (simulated) edges in basis points.
    realised_edge_bps:
        Sequence of realised edges in basis points, aligned with
        ``imagined_edge_bps``.
    minimum_correlation:
        Threshold that the Pearson correlation must meet or exceed.
    weights:
        Optional per-observation weights.  When omitted, all pairs are equally
        weighted.
    """

    if len(imagined_edge_bps) != len(realised_edge_bps):
        raise ValueError("imagined and realised edge sequences must be the same length")
    if not imagined_edge_bps:
        raise ValueError("edge sequences must not be empty")
    if weights is not None and len(weights) != len(imagined_edge_bps):
        raise ValueError("weights must align with edge sequences")

    threshold = _validate_threshold(minimum_correlation)

    cleaned_pairs: list[tuple[float, float, float]] = []
    total_pairs = len(imagined_edge_bps)

    for index, (imagined, realised) in enumerate(zip(imagined_edge_bps, realised_edge_bps)):
        raw_weight = 1.0 if weights is None else weights[index]
        if raw_weight is None:
            continue
        weight = _coerce_float(raw_weight, allow_none=False)
        if weight is None or weight <= 0.0 or not math.isfinite(weight):
            continue

        imagined_value = _coerce_float(imagined, allow_none=True)
        realised_value = _coerce_float(realised, allow_none=True)
        if imagined_value is None or realised_value is None:
            continue

        cleaned_pairs.append((imagined_value, realised_value, float(weight)))

    correlation = _weighted_correlation(cleaned_pairs)
    passed = correlation is not None and correlation >= threshold

    return PlannerEdgeGateDecision(
        passed=passed,
        correlation=correlation,
        minimum_correlation=threshold,
        valid_pairs=len(cleaned_pairs),
        total_pairs=total_pairs,
    )
