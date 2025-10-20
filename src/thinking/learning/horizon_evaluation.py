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
from typing import Iterable, Literal, Mapping, Sequence

import math

__all__ = [
    "HorizonObservation",
    "HorizonMetrics",
    "HorizonEvaluationReport",
    "evaluate_predictions_by_horizon",
    "HoldoutCalibrationResult",
    "calibrate_holdout_day",
]


_EVENT_ALIASES = {"event", "events", "event-time", "eventtime"}
_TIME_ALIASES = {"time", "temporal", "wall", "wall-time", "walltime", "wall-clock", "wallclock"}

_CANONICAL_EVENT_HORIZONS: tuple[str, ...] = ("ev1", "ev5", "ev20")
_CANONICAL_TIME_HORIZONS: tuple[str, ...] = ("100ms", "500ms", "2s")

_EVENT_HORIZON_ALIAS_MAP: dict[str, str] = {
    "1": "ev1",
    "ev1": "ev1",
    "5": "ev5",
    "ev5": "ev5",
    "20": "ev20",
    "ev20": "ev20",
}

_TIME_HORIZON_ALIAS_MAP: dict[str, str] = {
    "100ms": "100ms",
    "0.1s": "100ms",
    "0.10s": "100ms",
    "100": "100ms",
    "0.100s": "100ms",
    "500ms": "500ms",
    "0.5s": "500ms",
    "0.50s": "500ms",
    "500": "500ms",
    "2s": "2s",
    "2.0s": "2s",
    "2.00s": "2s",
    "2000ms": "2s",
}


def _normalise_horizon_type(value: str) -> str:
    normalised = value.strip().lower()
    if normalised in _EVENT_ALIASES:
        return "event"
    if normalised in _TIME_ALIASES:
        return "time"
    raise ValueError(f"unknown horizon_type '{value}'")


def _canonicalise_horizon_label(horizon: str | int | float, *, horizon_type: str) -> str:
    text = str(horizon).strip()
    if not text:
        return text

    key = text.lower()
    if horizon_type == "event":
        canonical = _EVENT_HORIZON_ALIAS_MAP.get(key)
        if canonical is not None:
            return canonical
        if key.isdigit():
            return f"ev{key}"
        try:
            numeric_value = float(text)
        except ValueError:
            return text
        if math.isfinite(numeric_value):
            rounded = round(numeric_value)
            if math.isclose(numeric_value, rounded, rel_tol=0.0, abs_tol=1e-12):
                integer_text = str(int(rounded))
                canonical = _EVENT_HORIZON_ALIAS_MAP.get(integer_text)
                if canonical is not None:
                    return canonical
                return f"ev{integer_text}"
        return text

    if horizon_type == "time":
        canonical = _TIME_HORIZON_ALIAS_MAP.get(key)
        if canonical is not None:
            return canonical
        try:
            numeric_value = float(text)
        except ValueError:
            return text
        if math.isfinite(numeric_value):
            if math.isclose(numeric_value, 0.1, rel_tol=0.0, abs_tol=1e-12):
                return "100ms"
            if math.isclose(numeric_value, 0.5, rel_tol=0.0, abs_tol=1e-12):
                return "500ms"
            if math.isclose(numeric_value, 2.0, rel_tol=0.0, abs_tol=1e-12):
                return "2s"
        return text

    return text


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
        canonical_horizon = _canonicalise_horizon_label(
            horizon, horizon_type=normalised_type
        )

        object.__setattr__(self, "horizon", canonical_horizon)
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


@dataclass(frozen=True, slots=True)
class HoldoutCalibrationResult:
    """Calibration summary for a hold-out evaluation window."""

    label: str | None
    method: str
    num_observations: int
    num_bins: int
    baseline_ece: float
    baseline_brier: float
    calibrated_ece: float
    calibrated_brier: float
    temperature: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "method": self.method,
            "num_observations": self.num_observations,
            "num_bins": self.num_bins,
            "baseline_ece": self.baseline_ece,
            "baseline_brier": self.baseline_brier,
            "calibrated_ece": self.calibrated_ece,
            "calibrated_brier": self.calibrated_brier,
            "temperature": self.temperature,
        }


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

    metrics_by_key: dict[tuple[str, str], HorizonMetrics] = {}
    for key in sorted(grouped, key=lambda item: order[item]):
        horizon_type, horizon = key
        metric = _compute_horizon_metrics(
            horizon=horizon,
            horizon_type=horizon_type,
            observations=grouped[key],
            num_bins=num_bins,
        )
        metrics_by_key[key] = metric

    event_metrics = _ordered_metrics_for_type(
        metrics_by_key,
        horizon_type="event",
        canonical_order=_CANONICAL_EVENT_HORIZONS,
        insertion_order=order,
    )
    time_metrics = _ordered_metrics_for_type(
        metrics_by_key,
        horizon_type="time",
        canonical_order=_CANONICAL_TIME_HORIZONS,
        insertion_order=order,
    )

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


def _ordered_metrics_for_type(
    metrics_by_key: Mapping[tuple[str, str], HorizonMetrics],
    *,
    horizon_type: str,
    canonical_order: Sequence[str],
    insertion_order: Mapping[tuple[str, str], int],
) -> tuple[HorizonMetrics, ...]:
    typed_keys = [key for key in metrics_by_key if key[0] == horizon_type]
    if not typed_keys:
        return tuple()

    ordered: list[HorizonMetrics] = []
    for label in canonical_order:
        key = (horizon_type, label)
        metric = metrics_by_key.get(key)
        if metric is not None:
            ordered.append(metric)

    fallback_keys = [
        key for key in typed_keys if key[1] not in canonical_order
    ]
    fallback_keys.sort(key=lambda item: insertion_order[item])
    ordered.extend(metrics_by_key[key] for key in fallback_keys)
    return tuple(ordered)


def _clip_probability(value: float, minimum: float = 1e-6, maximum: float = 1.0 - 1e-6) -> float:
    if value <= minimum:
        return minimum
    if value >= maximum:
        return maximum
    return value


def _safe_sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _apply_temperature_scaling(probabilities: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    scaled: list[float] = []
    for probability in probabilities:
        clipped = _clip_probability(probability)
        logit = math.log(clipped / (1.0 - clipped))
        scaled.append(_safe_sigmoid(logit / temperature))
    return scaled


def _fit_temperature_scaling(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    weights: Sequence[float],
    *,
    min_temperature: float = 0.25,
    max_temperature: float = 4.0,
    candidate_count: int = 121,
) -> tuple[float, list[float]]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if candidate_count < 3:
        raise ValueError("candidate_count must be at least 3")
    if min_temperature <= 0.0 or max_temperature <= 0.0:
        raise ValueError("temperature bounds must be positive")
    if max_temperature <= min_temperature:
        raise ValueError("max_temperature must exceed min_temperature")

    span = max_temperature - min_temperature
    step = span / (candidate_count - 1)

    best_temperature = 1.0
    best_loss = float("inf")

    for index in range(candidate_count):
        temperature = min_temperature + index * step
        if temperature < min_temperature:
            temperature = min_temperature
        elif temperature > max_temperature:
            temperature = max_temperature
        scaled = _apply_temperature_scaling(probabilities, temperature)
        loss = _weighted_brier_score(scaled, outcomes, weights)
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature

    refined_candidates = [best_temperature + offset * step for offset in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    for candidate in refined_candidates:
        if candidate <= 0.0:
            continue
        candidate = min(max(candidate, min_temperature), max_temperature)
        scaled = _apply_temperature_scaling(probabilities, candidate)
        loss = _weighted_brier_score(scaled, outcomes, weights)
        if loss < best_loss:
            best_loss = loss
            best_temperature = candidate

    calibrated = _apply_temperature_scaling(probabilities, best_temperature)
    return best_temperature, calibrated


def _fit_isotonic_regression(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    weights: Sequence[float],
) -> list[float]:
    if not probabilities:
        raise ValueError("probabilities must not be empty")

    order = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx])
    blocks: list[
        dict[str, object]
    ] = []

    for idx in order:
        probability = probabilities[idx]
        outcome = outcomes[idx]
        weight = weights[idx]
        block = {
            "min": probability,
            "max": probability,
            "weight": weight,
            "weighted_outcome": outcome * weight,
            "indices": [idx],
        }
        blocks.append(block)

        while len(blocks) >= 2:
            last = blocks[-1]
            prev = blocks[-2]
            last_weight = float(last["weight"])
            prev_weight = float(prev["weight"])
            if last_weight <= 0.0 or prev_weight <= 0.0:
                break
            last_avg = float(last["weighted_outcome"]) / last_weight
            prev_avg = float(prev["weighted_outcome"]) / prev_weight
            if prev_avg <= last_avg:
                break
            merged_weight = prev_weight + last_weight
            merged_block = {
                "min": float(prev["min"]),
                "max": float(last["max"]),
                "weight": merged_weight,
                "weighted_outcome": float(prev["weighted_outcome"]) + float(last["weighted_outcome"]),
                "indices": list(prev["indices"]) + list(last["indices"]),
            }
            blocks[-2] = merged_block
            blocks.pop()

    fitted = [0.0] * len(probabilities)
    for block in blocks:
        weight = float(block["weight"]) if float(block["weight"]) > 0.0 else 1.0
        value = float(block["weighted_outcome"]) / weight
        value = max(0.0, min(1.0, value))
        for idx in block["indices"]:  # type: ignore[index]
            fitted[int(idx)] = value
    return fitted


def calibrate_holdout_day(
    observations: Sequence[HorizonObservation],
    *,
    method: Literal["temperature", "isotonic"] = "temperature",
    num_bins: int = 10,
    label: str | None = None,
    temperature_bounds: tuple[float, float] = (0.25, 4.0),
    temperature_candidates: int = 121,
) -> HoldoutCalibrationResult:
    """Calibrate a hold-out day and report ECE and Brier scores.

    The calibration operates directly on probability forecasts attached to the
    provided observations.  Temperature scaling optimises a single scalar
    parameter to minimise the weighted Brier loss, while isotonic regression
    fits a monotonic stepwise function.  In both cases, the outcome and weight
    inputs are preserved and only the probabilities are transformed.
    """

    if not observations:
        raise ValueError("observations must not be empty")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    method_key = method.lower()
    if method_key not in {"temperature", "isotonic"}:
        raise ValueError("method must be 'temperature' or 'isotonic'")

    probabilities = [obs.probability for obs in observations]
    outcomes = [obs.outcome for obs in observations]
    weights = [obs.weight for obs in observations]

    baseline_ece = _expected_calibration_error(probabilities, outcomes, weights, num_bins=num_bins)
    baseline_brier = _weighted_brier_score(probabilities, outcomes, weights)

    if method_key == "temperature":
        min_temp, max_temp = temperature_bounds
        temperature, calibrated = _fit_temperature_scaling(
            probabilities,
            outcomes,
            weights,
            min_temperature=min_temp,
            max_temperature=max_temp,
            candidate_count=temperature_candidates,
        )
    else:
        temperature = None
        calibrated = _fit_isotonic_regression(probabilities, outcomes, weights)

    calibrated_ece = _expected_calibration_error(calibrated, outcomes, weights, num_bins=num_bins)
    calibrated_brier = _weighted_brier_score(calibrated, outcomes, weights)

    return HoldoutCalibrationResult(
        label=label,
        method=method_key,
        num_observations=len(observations),
        num_bins=num_bins,
        baseline_ece=baseline_ece,
        baseline_brier=baseline_brier,
        calibrated_ece=calibrated_ece,
        calibrated_brier=calibrated_brier,
        temperature=temperature,
    )
