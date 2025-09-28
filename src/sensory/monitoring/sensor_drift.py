"""Sensor drift detection harness for roadmap Phase 2 monitoring.

The high-impact roadmap calls for a lightweight anomaly detection harness that
flags sensory drifts before they cascade into trading logic.  This module keeps
the implementation dependency-light while exposing well-typed artefacts that
strategies, risk modules, and operational tooling can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SensorDriftBaseline",
    "SensorDriftParameters",
    "SensorDriftResult",
    "SensorDriftSummary",
    "evaluate_sensor_drift",
]


@dataclass(slots=True)
class SensorDriftBaseline:
    """Summary statistics representing an expected sensory window."""

    sensor: str
    mean: float
    std: float
    count: int


@dataclass(slots=True)
class SensorDriftParameters:
    """Configuration values applied during drift evaluation."""

    baseline_window: int
    evaluation_window: int
    min_observations: int
    z_threshold: float


@dataclass(slots=True)
class SensorDriftResult:
    """Calculated drift characteristics for a single sensor."""

    sensor: str
    baseline: SensorDriftBaseline
    evaluation_mean: float
    evaluation_std: float
    evaluation_count: int
    z_score: float | None
    drift_ratio: float
    exceeded: bool

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-friendly representation."""

        return {
            "sensor": self.sensor,
            "baseline_mean": self.baseline.mean,
            "baseline_std": self.baseline.std,
            "baseline_count": self.baseline.count,
            "evaluation_mean": self.evaluation_mean,
            "evaluation_std": self.evaluation_std,
            "evaluation_count": self.evaluation_count,
            "z_score": self.z_score,
            "drift_ratio": self.drift_ratio,
            "exceeded": self.exceeded,
        }


@dataclass(slots=True)
class SensorDriftSummary:
    """Aggregate view of drift results for downstream tooling."""

    parameters: SensorDriftParameters
    results: tuple[SensorDriftResult, ...]

    @property
    def exceeded(self) -> tuple[SensorDriftResult, ...]:
        """Return all sensors whose |z-score| breached the configured threshold."""

        return tuple(result for result in self.results if result.exceeded)

    def as_dict(self) -> Mapping[str, object]:
        """Serialise the summary and underlying results."""

        return {
            "parameters": {
                "baseline_window": self.parameters.baseline_window,
                "evaluation_window": self.parameters.evaluation_window,
                "min_observations": self.parameters.min_observations,
                "z_threshold": self.parameters.z_threshold,
            },
            "results": [result.as_dict() for result in self.results],
        }


def _coerce_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return values.to_numpy(dtype=float, copy=False)


def _compute_baseline(sensor: str, window: np.ndarray) -> SensorDriftBaseline | None:
    if window.size == 0:
        return None
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=1 if window.size > 1 else 0))
    return SensorDriftBaseline(sensor=sensor, mean=mean, std=std, count=int(window.size))


def _standard_error(baseline: SensorDriftBaseline, eval_std: float, eval_count: int) -> float | None:
    if baseline.count == 0 or eval_count == 0:
        return None
    baseline_var = baseline.std ** 2 if baseline.count > 1 else 0.0
    eval_var = eval_std ** 2 if eval_count > 1 else 0.0
    se = sqrt((baseline_var / max(baseline.count, 1)) + (eval_var / max(eval_count, 1)))
    if se == 0.0:
        return None
    return se


def _relative_drift(baseline: SensorDriftBaseline, evaluation_mean: float) -> float:
    denominator = abs(baseline.mean) if abs(baseline.mean) > 1e-9 else 1.0
    return abs(evaluation_mean - baseline.mean) / denominator


def _build_result(
    sensor: str,
    baseline: SensorDriftBaseline,
    evaluation_values: np.ndarray,
    z_threshold: float,
) -> SensorDriftResult | None:
    if evaluation_values.size == 0:
        return None

    evaluation_mean = float(np.mean(evaluation_values))
    evaluation_std = float(np.std(evaluation_values, ddof=1 if evaluation_values.size > 1 else 0))
    evaluation_count = int(evaluation_values.size)

    se = _standard_error(baseline, evaluation_std, evaluation_count)
    z_score: float | None
    exceeded = False
    if se is None:
        z_score = None
    else:
        z_score = (evaluation_mean - baseline.mean) / se
        exceeded = abs(z_score) >= z_threshold

    drift_ratio = _relative_drift(baseline, evaluation_mean)

    return SensorDriftResult(
        sensor=sensor,
        baseline=baseline,
        evaluation_mean=evaluation_mean,
        evaluation_std=evaluation_std,
        evaluation_count=evaluation_count,
        z_score=None if z_score is None else float(z_score),
        drift_ratio=drift_ratio,
        exceeded=exceeded,
    )


def evaluate_sensor_drift(
    frame: pd.DataFrame,
    *,
    sensor_columns: Sequence[str] | None = None,
    baseline_window: int = 240,
    evaluation_window: int = 60,
    min_observations: int = 20,
    z_threshold: float = 3.0,
) -> SensorDriftSummary:
    """Evaluate sensory drift over trailing windows.

    Args:
        frame: DataFrame containing sensory readings ordered chronologically.
        sensor_columns: Optional subset of columns to analyse. Defaults to all
            numeric columns in ``frame``.
        baseline_window: Number of rows constituting the reference window.
        evaluation_window: Number of most recent rows used for drift detection.
        min_observations: Minimum observations required for both windows before
            computing drift.
        z_threshold: Absolute z-score threshold for flagging drift.
    """

    if baseline_window <= 0 or evaluation_window <= 0:
        raise ValueError("baseline_window and evaluation_window must be positive")
    if frame.empty:
        raise ValueError("frame must contain observations")
    if frame.shape[0] < baseline_window + evaluation_window:
        raise ValueError("insufficient rows for the requested windows")

    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    if sensor_columns is None:
        candidate_columns = numeric_columns
    else:
        candidate_columns = [col for col in sensor_columns if col in frame.columns]

    if not candidate_columns:
        raise ValueError("no sensor columns available for drift evaluation")

    baseline_slice = frame.iloc[-(baseline_window + evaluation_window) : -evaluation_window]
    evaluation_slice = frame.iloc[-evaluation_window:]

    parameters = SensorDriftParameters(
        baseline_window=baseline_window,
        evaluation_window=evaluation_window,
        min_observations=min_observations,
        z_threshold=z_threshold,
    )

    results: list[SensorDriftResult] = []
    for column in candidate_columns:
        baseline_values = _coerce_numeric(baseline_slice[column])
        evaluation_values = _coerce_numeric(evaluation_slice[column])
        if baseline_values.size < min_observations or evaluation_values.size < min_observations:
            continue
        baseline = _compute_baseline(column, baseline_values)
        if baseline is None:
            continue
        result = _build_result(column, baseline, evaluation_values, z_threshold)
        if result is not None:
            results.append(result)

    results.sort(key=lambda item: (abs(item.z_score) if item.z_score is not None else 0.0), reverse=True)
    return SensorDriftSummary(parameters=parameters, results=tuple(results))
