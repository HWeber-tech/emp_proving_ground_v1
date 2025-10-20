"""Horizon calibration metrics for the LOBSTER pre-training roadmap (E.1.2).

This module evaluates probabilistic forecasts bucketed by horizon categories
(``events`` and ``time``) and reports:

* Expected Calibration Error (ECE)
* Brier score
* Net alpha after fees (in basis points)

It provides lightweight, dependency-free helpers so the evaluation can run in
CI and during research without relying on pandas/NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math

__all__ = [
    "HorizonObservation",
    "HorizonMetrics",
    "HorizonEvaluationReport",
    "evaluate_predictions_by_horizon",
]


_EVENT_ALIASES = {"event", "events"}
_TIME_ALIASES = {"time", "temporal"}


def _normalise_horizon_type(value: str) -> str:
    normalised = value.strip().lower()
    if normalised in _EVENT_ALIASES:
        return "event"
    if normalised in _TIME_ALIASES:
        return "time"
    raise ValueError(f"unknown horizon_type '{value}'")


def _coerce_probability(value: float) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability must be a finite float") from exc
    if not math.isfinite(probability):
        raise ValueError("probability must be a finite float")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must fall within [0, 1]")
    return probability


def _coerce_outcome(value: bool | float | int) -> float:
    try:
        outcome = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be coercible to a float") from exc
    if not math.isfinite(outcome):
        raise ValueError("outcome must be a finite float")
    if outcome < 0.0 or outcome > 1.0:
        raise ValueError("outcome must fall within [0, 1]")
    return outcome


def _coerce_float(value: float, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be coercible to a float") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _coerce_weight(value: float) -> float:
    numeric = _coerce_float(value, field="weight")
    if numeric <= 0.0:
        raise ValueError("weight must be strictly positive")
    return numeric


@dataclass(frozen=True, slots=True)
class HorizonObservation:
    """Single probabilistic forecast observation for a given horizon."""

    horizon: str
    horizon_type: str
    probability: float
    outcome: float
    gross_alpha_bps: float
    fees_bps: float
    weight: float

    def __init__(
        self,
        horizon: str | int | float,
        horizon_type: str,
        *,
        probability: float,
        outcome: bool | float | int,
        gross_alpha_bps: float = 0.0,
        fees_bps: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        normalised_type = _normalise_horizon_type(horizon_type)
        coerced_probability = _coerce_probability(probability)
        coerced_outcome = _coerce_outcome(outcome)
        coerced_gross_alpha = _coerce_float(gross_alpha_bps, field="gross_alpha_bps")
        coerced_fees = _coerce_float(fees_bps, field="fees_bps")
        coerced_weight = _coerce_weight(weight)

        object.__setattr__(self, "horizon", str(horizon))
        object.__setattr__(self, "horizon_type", normalised_type)
        object.__setattr__(self, "probability", coerced_probability)
        object.__setattr__(self, "outcome", coerced_outcome)
        object.__setattr__(self, "gross_alpha_bps", coerced_gross_alpha)
        object.__setattr__(self, "fees_bps", coerced_fees)
        object.__setattr__(self, "weight", coerced_weight)


@dataclass(frozen=True, slots=True)
class HorizonMetrics:
    """Aggregated metrics for a particular horizon bucket."""

    horizon: str
    horizon_type: str
    count: int
    total_weight: float
    ece: float
    brier: float
    alpha_after_fees_bps: float
    gross_alpha_bps: float
    fees_bps: float
    average_probability: float
    average_outcome: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "horizon": self.horizon,
            "horizon_type": self.horizon_type,
            "count": self.count,
            "total_weight": self.total_weight,
            "ece": self.ece,
            "brier": self.brier,
            "alpha_after_fees_bps": self.alpha_after_fees_bps,
            "gross_alpha_bps": self.gross_alpha_bps,
            "fees_bps": self.fees_bps,
            "average_probability": self.average_probability,
            "average_outcome": self.average_outcome,
        }


@dataclass(frozen=True, slots=True)
class HorizonEvaluationReport:
    """Summary of calibration metrics across event/time horizons."""

    event_horizons: tuple[HorizonMetrics, ...]
    time_horizons: tuple[HorizonMetrics, ...]
    overall: HorizonMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "event_horizons": [metric.as_dict() for metric in self.event_horizons],
            "time_horizons": [metric.as_dict() for metric in self.time_horizons],
            "overall": self.overall.as_dict(),
        }

    @property
    def by_horizon(self) -> tuple[HorizonMetrics, ...]:
        return self.event_horizons + self.time_horizons


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        total += value * weight
        weight_sum += weight
    if weight_sum == 0.0:
        raise ValueError("weights must not sum to zero")
    return total / weight_sum


def _weighted_brier_score(
    probabilities: Sequence[float], outcomes: Sequence[float], weights: Sequence[float]
) -> float:
    total = 0.0
    weight_sum = 0.0
    for probability, outcome, weight in zip(probabilities, outcomes, weights):
        diff = probability - outcome
        total += weight * diff * diff
        weight_sum += weight
    if weight_sum == 0.0:
        return 0.0
    return total / weight_sum


def _expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    weights: Sequence[float],
    *,
    num_bins: int,
) -> float:
    bin_totals = [0.0] * num_bins
    bin_probabilities = [0.0] * num_bins
    bin_outcomes = [0.0] * num_bins

    for probability, outcome, weight in zip(probabilities, outcomes, weights):
        index = min(int(probability * num_bins), num_bins - 1)
        bin_totals[index] += weight
        bin_probabilities[index] += probability * weight
        bin_outcomes[index] += outcome * weight

    total_weight = sum(bin_totals)
    if total_weight == 0.0:
        return 0.0

    ece = 0.0
    for total, prob_sum, outcome_sum in zip(bin_totals, bin_probabilities, bin_outcomes):
        if total == 0.0:
            continue
        avg_probability = prob_sum / total
        avg_outcome = outcome_sum / total
        ece += (total / total_weight) * abs(avg_probability - avg_outcome)
    return ece


def _compute_horizon_metrics(
    *,
    horizon: str,
    horizon_type: str,
    observations: Sequence[HorizonObservation],
    num_bins: int,
) -> HorizonMetrics:
    probabilities = [obs.probability for obs in observations]
    outcomes = [obs.outcome for obs in observations]
    weights = [obs.weight for obs in observations]
    gross_alpha = [obs.gross_alpha_bps for obs in observations]
    fees = [obs.fees_bps for obs in observations]
    net_alpha = [g - f for g, f in zip(gross_alpha, fees)]

    total_weight = sum(weights)
    average_probability = _weighted_mean(probabilities, weights)
    average_outcome = _weighted_mean(outcomes, weights)
    average_gross_alpha = _weighted_mean(gross_alpha, weights)
    average_fees = _weighted_mean(fees, weights)
    average_net_alpha = _weighted_mean(net_alpha, weights)

    ece = _expected_calibration_error(probabilities, outcomes, weights, num_bins=num_bins)
    brier = _weighted_brier_score(probabilities, outcomes, weights)

    return HorizonMetrics(
        horizon=horizon,
        horizon_type=horizon_type,
        count=len(observations),
        total_weight=total_weight,
        ece=ece,
        brier=brier,
        alpha_after_fees_bps=average_net_alpha,
        gross_alpha_bps=average_gross_alpha,
        fees_bps=average_fees,
        average_probability=average_probability,
        average_outcome=average_outcome,
    )


def evaluate_predictions_by_horizon(
    observations: Sequence[HorizonObservation], *, num_bins: int = 10
) -> HorizonEvaluationReport:
    """Evaluate calibration metrics partitioned by event/time horizons."""

    if not observations:
        raise ValueError("observations must not be empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    grouped: dict[tuple[str, str], list[HorizonObservation]] = {}
    order: dict[tuple[str, str], int] = {}

    for index, observation in enumerate(observations):
        key = (observation.horizon_type, observation.horizon)
        grouped.setdefault(key, []).append(observation)
        order.setdefault(key, index)

    metrics: list[HorizonMetrics] = []
    for key in sorted(grouped, key=lambda item: order[item]):
        horizon_type, horizon = key
        metrics.append(
            _compute_horizon_metrics(
                horizon=horizon,
                horizon_type=horizon_type,
                observations=grouped[key],
                num_bins=num_bins,
            )
        )

    event_metrics = tuple(metric for metric in metrics if metric.horizon_type == "event")
    time_metrics = tuple(metric for metric in metrics if metric.horizon_type == "time")

    overall = _compute_horizon_metrics(
        horizon="all",
        horizon_type="all",
        observations=observations,
        num_bins=num_bins,
    )

    return HorizonEvaluationReport(
        event_horizons=event_metrics,
        time_horizons=time_metrics,
        overall=overall,
    )

