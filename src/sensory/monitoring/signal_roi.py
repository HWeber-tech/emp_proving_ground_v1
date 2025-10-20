"""Signal ROI monitor: quantify marginal predictive value of data streams.

The roadmap calls for instrumentation that ranks each sensory data stream by the
incremental predictive power it contributes to downstream models.  This module
implements a lightweight ridge-regression based attribution: it measures the
change in explanatory power when each stream is removed from a linear model and
reports the marginal lift in mean-squared-error and R-squared.

The implementation intentionally mirrors the design philosophy used elsewhere in
``src.sensory.monitoring``: dependency-light analytics with structured data
classes that can be exported to dashboards or observability pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "SignalRoiContribution",
    "SignalRoiSummary",
    "SignalRoiMonitor",
    "evaluate_signal_roi",
]


@dataclass(slots=True)
class SignalRoiContribution:
    """Marginal predictive value metrics for a single data stream."""

    stream: str
    marginal_r2: float
    r2_without: float
    marginal_mse_gain: float
    coefficient: float
    mean_value: float
    correlation: float | None
    t_stat: float | None
    share_of_gain: float | None

    def as_dict(self) -> dict[str, float | None | str]:
        return {
            "stream": self.stream,
            "marginal_r2": self.marginal_r2,
            "r2_without": self.r2_without,
            "marginal_mse_gain": self.marginal_mse_gain,
            "coefficient": self.coefficient,
            "mean_value": self.mean_value,
            "correlation": self.correlation,
            "t_stat": self.t_stat,
            "share_of_gain": self.share_of_gain,
        }


@dataclass(slots=True)
class SignalRoiSummary:
    """Aggregate report describing signal ROI posture."""

    samples: int
    r_squared: float
    baseline_mse: float
    model_mse: float
    roi_uplift: float
    intercept: float
    regularisation: float
    contributions: tuple[SignalRoiContribution, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "r_squared": self.r_squared,
            "baseline_mse": self.baseline_mse,
            "model_mse": self.model_mse,
            "roi_uplift": self.roi_uplift,
            "intercept": self.intercept,
            "regularisation": self.regularisation,
            "contributions": [contrib.as_dict() for contrib in self.contributions],
        }

    def to_dataframe(self):  # pragma: no cover - thin convenience wrapper
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("pandas is required to convert ROI contributions to a DataFrame") from exc
        return pd.DataFrame(contrib.as_dict() for contrib in self.contributions)


class SignalRoiMonitor:
    """Convenience wrapper for evaluating marginal predictive value."""

    def __init__(self, *, regularisation: float = 1e-6) -> None:
        if regularisation < 0:
            raise ValueError("regularisation must be non-negative")
        self._regularisation = float(regularisation)

    @property
    def regularisation(self) -> float:
        return self._regularisation

    def evaluate(
        self,
        target: Sequence[float] | np.ndarray,
        data_streams: Mapping[str, Sequence[float] | np.ndarray],
    ) -> SignalRoiSummary:
        return evaluate_signal_roi(
            target,
            data_streams,
            regularisation=self._regularisation,
        )


def evaluate_signal_roi(
    target: Sequence[float] | np.ndarray,
    data_streams: Mapping[str, Sequence[float] | np.ndarray],
    *,
    regularisation: float = 1e-6,
) -> SignalRoiSummary:
    """Compute marginal predictive value for each data stream via ridge regression."""

    if regularisation < 0:
        raise ValueError("regularisation must be non-negative")
    if not data_streams:
        raise ValueError("data_streams must contain at least one entry")

    y = np.asarray(target, dtype=float)
    if y.ndim != 1:
        raise ValueError("target must be one-dimensional")

    streams: list[np.ndarray] = []
    names: list[str] = []
    for name, values in data_streams.items():
        names.append(str(name))
        streams.append(np.asarray(values, dtype=float))

    if any(stream.ndim != 1 for stream in streams):
        raise ValueError("all data streams must be one-dimensional sequences")

    lengths = {stream.shape[0] for stream in streams}
    lengths.add(y.shape[0])
    if len(lengths) != 1:
        raise ValueError("target and all data streams must have the same length")

    mask = np.isfinite(y)
    for stream in streams:
        mask &= np.isfinite(stream)

    y_clean = y[mask]
    if y_clean.size == 0:
        raise ValueError("no valid observations after dropping NaNs")

    X = np.column_stack([stream[mask] for stream in streams])
    coefficients, intercept, covariance = _ridge_regression(
        y_clean,
        X,
        regularisation=regularisation,
    )

    predictions = X @ coefficients + intercept
    residuals = y_clean - predictions

    samples = int(y_clean.size)
    y_mean = float(np.mean(y_clean))
    ss_total = float(np.sum((y_clean - y_mean) ** 2))
    ss_residual = float(np.sum(residuals**2))

    r_squared = 0.0 if ss_total == 0 else 1.0 - ss_residual / ss_total
    baseline_mse = 0.0 if samples == 0 else ss_total / samples
    model_mse = 0.0 if samples == 0 else ss_residual / samples
    roi_uplift = baseline_mse - model_mse

    raw_contributions: list[dict[str, float | None | str]] = []
    positive_gain_sum = 0.0

    for idx, name in enumerate(names):
        subX = np.delete(X, idx, axis=1)
        coeffs_without, intercept_without, _ = _ridge_regression(
            y_clean,
            subX,
            regularisation=regularisation,
        )
        predictions_without = subX @ coeffs_without + intercept_without
        residuals_without = y_clean - predictions_without
        ss_res_without = float(np.sum(residuals_without**2))
        r2_without = 0.0 if ss_total == 0 else 1.0 - ss_res_without / ss_total
        marginal_r2 = r_squared - r2_without
        marginal_mse_gain = (ss_res_without - ss_residual) / samples if samples else 0.0
        if marginal_mse_gain > 0:
            positive_gain_sum += marginal_mse_gain

        stream_values = X[:, idx]
        mean_value = float(np.mean(stream_values))

        if stream_values.size > 1 and np.std(stream_values) > 0 and np.std(y_clean) > 0:
            correlation = float(np.corrcoef(stream_values, y_clean)[0, 1])
        else:
            correlation = None

        if covariance.size == 0:
            t_stat = None
        else:
            variance = float(covariance[idx, idx]) if covariance.shape[0] > idx else 0.0
            if variance <= 0.0:
                t_stat = None
            else:
                t_stat = float(coefficients[idx] / np.sqrt(variance))

        raw_contributions.append(
            {
                "stream": name,
                "marginal_r2": float(marginal_r2),
                "r2_without": float(r2_without),
                "marginal_mse_gain": float(marginal_mse_gain),
                "coefficient": float(coefficients[idx]),
                "mean_value": mean_value,
                "correlation": correlation,
                "t_stat": t_stat,
            }
        )

    raw_contributions.sort(key=lambda payload: payload["marginal_mse_gain"], reverse=True)

    contributions: list[SignalRoiContribution] = []
    for payload in raw_contributions:
        gain = float(payload["marginal_mse_gain"])  # type: ignore[arg-type]
        if positive_gain_sum > 0 and gain > 0:
            share = gain / positive_gain_sum
        else:
            share = None
        contributions.append(
            SignalRoiContribution(
                stream=str(payload["stream"]),
                marginal_r2=float(payload["marginal_r2"]),
                r2_without=float(payload["r2_without"]),
                marginal_mse_gain=gain,
                coefficient=float(payload["coefficient"]),
                mean_value=float(payload["mean_value"]),
                correlation=(
                    None
                    if payload["correlation"] is None
                    else float(payload["correlation"])
                ),
                t_stat=(
                    None
                    if payload["t_stat"] is None
                    else float(payload["t_stat"])
                ),
                share_of_gain=share,
            )
        )

    return SignalRoiSummary(
        samples=samples,
        r_squared=float(r_squared),
        baseline_mse=float(baseline_mse),
        model_mse=float(model_mse),
        roi_uplift=float(roi_uplift),
        intercept=float(intercept),
        regularisation=float(regularisation),
        contributions=tuple(contributions),
    )


def _ridge_regression(
    y: np.ndarray,
    X: np.ndarray,
    *,
    regularisation: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve a ridge regression returning coefficients, intercept, covariance."""

    if y.ndim != 1:
        raise ValueError("target array must be one-dimensional")
    if X.ndim != 2:
        raise ValueError("design matrix must be two-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError("design matrix row count must match target length")

    n, k = X.shape
    if n == 0:
        raise ValueError("ridge regression requires at least one observation")
    if regularisation < 0:
        raise ValueError("regularisation must be non-negative")

    design = np.column_stack([X, np.ones(n)])
    identity = np.eye(k + 1)
    identity[-1, -1] = 0.0  # do not regularise the intercept
    gram = design.T @ design + regularisation * identity

    try:
        inv_gram = np.linalg.inv(gram)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive guard
        raise ValueError("design matrix is singular") from exc

    beta = inv_gram @ design.T @ y
    coefficients = beta[:-1]
    intercept = float(beta[-1])
    residuals = y - design @ beta
    dof = max(n - (k + 1), 1)
    sigma2 = float(residuals.T @ residuals) / dof
    covariance = inv_gram * sigma2
    return coefficients, intercept, covariance
