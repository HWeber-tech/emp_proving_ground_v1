"""A/B validation helpers for the Tiny Recursive Model surrogate simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class SurrogateABValidationResult:
    """Outcome of comparing surrogate simulation results against ground truth."""

    metric_name: str
    surrogate_metric: float
    ground_truth_metric: float
    surrogate_turnover: float
    ground_truth_turnover: float
    metric_diff_pct: float
    turnover_diff_pct: float
    metric_within_threshold: bool
    turnover_within_threshold: bool
    should_retrain: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> Mapping[str, object]:
        """Return a JSON-serialisable payload for reporting."""

        return {
            "metric_name": self.metric_name,
            "surrogate_metric": self.surrogate_metric,
            "ground_truth_metric": self.ground_truth_metric,
            "surrogate_turnover": self.surrogate_turnover,
            "ground_truth_turnover": self.ground_truth_turnover,
            "metric_diff_pct": self.metric_diff_pct,
            "turnover_diff_pct": self.turnover_diff_pct,
            "metric_within_threshold": self.metric_within_threshold,
            "turnover_within_threshold": self.turnover_within_threshold,
            "should_retrain": self.should_retrain,
            "notes": list(self.notes),
        }


def validate_surrogate_alignment(
    surrogate_metrics: Mapping[str, Any] | Any,
    ground_truth_metrics: Mapping[str, Any] | Any,
    *,
    metric: str = "sharpe",
    metric_tolerance_pct: float = 5.0,
    turnover_tolerance_pct: float = 10.0,
) -> SurrogateABValidationResult:
    """Compare surrogate metrics against ground-truth simulation outputs.

    Parameters
    ----------
    surrogate_metrics:
        Metrics produced by the surrogate simulator. Must expose both the target
        ``metric`` (default ``"sharpe"``) and ``turnover`` either via mapping
        access or as attributes.
    ground_truth_metrics:
        Metrics produced by the authoritative simulator under the same
        conditions. Must expose the same keys or attributes as
        ``surrogate_metrics``.
    metric:
        Name of the primary performance metric to compare. The default matches
        the roadmap gating language which monitors Sharpe deltas.
    metric_tolerance_pct:
        Maximum allowed absolute percentage difference between surrogate and
        ground truth for the primary metric before retraining is triggered.
    turnover_tolerance_pct:
        Maximum allowed absolute percentage difference for turnover before
        retraining is triggered.

    Returns
    -------
    SurrogateABValidationResult
        Structured comparison including per-metric percentage deltas and a
        ``should_retrain`` flag when thresholds are breached.
    """

    metric_key = str(metric).strip()
    if not metric_key:
        raise ValueError("metric name must be a non-empty string")
    if metric_tolerance_pct < 0:
        raise ValueError("metric_tolerance_pct must be non-negative")
    if turnover_tolerance_pct < 0:
        raise ValueError("turnover_tolerance_pct must be non-negative")

    surrogate_metric = _extract_metric(surrogate_metrics, metric_key, context="surrogate")
    ground_truth_metric = _extract_metric(ground_truth_metrics, metric_key, context="ground_truth")
    surrogate_turnover = _extract_metric(surrogate_metrics, "turnover", context="surrogate")
    ground_truth_turnover = _extract_metric(ground_truth_metrics, "turnover", context="ground_truth")

    metric_diff_pct = _relative_difference_pct(surrogate_metric, ground_truth_metric)
    turnover_diff_pct = _relative_difference_pct(surrogate_turnover, ground_truth_turnover)

    metric_within_threshold = metric_diff_pct <= metric_tolerance_pct + 1e-9
    turnover_within_threshold = turnover_diff_pct <= turnover_tolerance_pct + 1e-9
    should_retrain = not (metric_within_threshold and turnover_within_threshold)

    notes: list[str] = []
    if not metric_within_threshold:
        notes.append(
            f"{metric_key} diff {metric_diff_pct:.2f}% exceeds tolerance {metric_tolerance_pct:.2f}%"
        )
    if not turnover_within_threshold:
        notes.append(
            f"turnover diff {turnover_diff_pct:.2f}% exceeds tolerance {turnover_tolerance_pct:.2f}%"
        )

    return SurrogateABValidationResult(
        metric_name=metric_key,
        surrogate_metric=surrogate_metric,
        ground_truth_metric=ground_truth_metric,
        surrogate_turnover=surrogate_turnover,
        ground_truth_turnover=ground_truth_turnover,
        metric_diff_pct=metric_diff_pct,
        turnover_diff_pct=turnover_diff_pct,
        metric_within_threshold=metric_within_threshold,
        turnover_within_threshold=turnover_within_threshold,
        should_retrain=should_retrain,
        notes=tuple(notes),
    )


def _extract_metric(metrics: Mapping[str, Any] | Any, key: str, *, context: str) -> float:
    """Return the metric value from mappings or attribute containers."""

    value: Any
    if isinstance(metrics, Mapping):
        if key not in metrics:
            raise KeyError(f"{context} metrics missing '{key}'")
        value = metrics[key]
    else:
        if not hasattr(metrics, key):
            raise KeyError(f"{context} metrics missing '{key}' attribute")
        value = getattr(metrics, key)
    return _coerce_float(value, field=f"{context}.{key}")


def _coerce_float(value: Any, *, field: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"{field} must be a float-compatible value") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"{field} must be finite")
    return coerced


def _relative_difference_pct(candidate: float, reference: float) -> float:
    """Return the absolute percentage difference using the reference as baseline."""

    baseline = abs(reference)
    if baseline < 1e-9:
        baseline = max(abs(candidate), 1.0)
    difference = abs(candidate - reference)
    if baseline == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / baseline * 100.0


__all__ = [
    "SurrogateABValidationResult",
    "validate_surrogate_alignment",
]
