"""Performance attribution utilities for roadmap Phase 2 alpha operations.

This module provides a light-weight feature attribution pipeline that converts
strategy backtest outputs into interpretable diagnostics.  It intentionally
avoids external dependencies beyond ``numpy`` so it can run inside CI during
nightly backtests while remaining easy to extend for richer analytics.

The implementation follows a pragmatic linear attribution approach:

* Align return series with feature exposure time-series supplied by
  strategies (e.g. momentum score, volatility estimate, risk budget).
* Perform a ridge-regularised least-squares regression to estimate the
  marginal contribution of each feature to realised returns.
* Derive summary statistics (Sharpe-like ratio, t-stats, residual drift)
  and expose helpers for producing DataFrame-friendly payloads.

Although simplified, the workflow mirrors the encyclopedia guidance for
"Alpha Ops" diagnostics and provides a foundation for richer explainability
work in later roadmap phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "FeatureContribution",
    "AttributionResult",
    "compute_performance_attribution",
    "result_to_dataframe",
]


@dataclass(slots=True)
class FeatureContribution:
    """Represents the marginal contribution of a feature to strategy returns."""

    name: str
    coefficient: float
    mean_exposure: float
    contribution: float
    t_stat: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "name": self.name,
            "coefficient": float(self.coefficient),
            "mean_exposure": float(self.mean_exposure),
            "contribution": float(self.contribution),
            "t_stat": float(self.t_stat) if self.t_stat is not None else None,
        }


@dataclass(slots=True)
class AttributionResult:
    """Container describing the outcome of a performance attribution run."""

    total_return: float
    average_return: float
    total_volatility: float
    sharpe_ratio: float | None
    residual_mean: float
    r_squared: float
    intercept: float
    contributions: tuple[FeatureContribution, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_return": self.total_return,
            "average_return": self.average_return,
            "total_volatility": self.total_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "residual_mean": self.residual_mean,
            "r_squared": self.r_squared,
            "intercept": self.intercept,
            "contributions": [contrib.as_dict() for contrib in self.contributions],
        }


class _AlignedArrays:
    """Helper structure bundling aligned returns and feature exposures."""

    def __init__(
        self,
        returns: Sequence[float] | np.ndarray,
        features: Mapping[str, Sequence[float] | np.ndarray],
    ) -> None:
        self.names = tuple(features.keys())
        self.returns, self.exposures = self._align(returns, features)

    @staticmethod
    def _align(
        returns: Sequence[float] | np.ndarray,
        features: Mapping[str, Sequence[float] | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        returns_array = np.asarray(returns, dtype=float)
        if returns_array.ndim != 1:
            raise ValueError("returns must be a one-dimensional sequence")
        if returns_array.size == 0:
            raise ValueError("returns must contain at least one observation")

        exposures: list[np.ndarray] = []
        valid_mask = np.isfinite(returns_array)
        for name, series in features.items():
            series_array = np.asarray(series, dtype=float)
            if series_array.shape != returns_array.shape:
                raise ValueError(
                    f"feature '{name}' must have the same length as returns"
                )
            valid_mask &= np.isfinite(series_array)
            exposures.append(series_array)

        returns_array = returns_array[valid_mask]
        if returns_array.size == 0:
            raise ValueError("no valid observations after dropping NaNs")

        exposure_matrix = np.vstack(exposures).T[valid_mask]
        return returns_array, exposure_matrix


def _ridge_regression(
    y: np.ndarray, X: np.ndarray, *, regularisation: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve a ridge regression and return coefficients, intercept, covariance."""

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if len(y) != len(X):
        raise ValueError("X and y must contain the same number of observations")

    n, k = X.shape
    design = np.column_stack([X, np.ones(n)])
    identity = np.eye(k + 1)
    identity[-1, -1] = 0.0  # do not regularise the intercept
    gram = design.T @ design + regularisation * identity
    try:
        inv_gram = np.linalg.inv(gram)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError("design matrix is singular") from exc

    beta = inv_gram @ design.T @ y
    coefficients = beta[:-1]
    intercept = float(beta[-1])
    residuals = y - design @ beta
    dof = max(n - (k + 1), 1)
    sigma2 = float(residuals.T @ residuals) / dof
    covariance = inv_gram * sigma2
    return coefficients, intercept, covariance


def compute_performance_attribution(
    returns: Sequence[float] | np.ndarray,
    features: Mapping[str, Sequence[float] | np.ndarray],
    *,
    regularisation: float = 1e-6,
) -> AttributionResult:
    """Estimate feature contributions to strategy returns.

    Args:
        returns: Sequence of realised returns for the strategy.
        features: Mapping of feature name -> exposure sequence.
        regularisation: Ridge penalty applied to stabilise the regression.

    Returns:
        AttributionResult summarising aggregate statistics and contributions.
    """

    if regularisation < 0:
        raise ValueError("regularisation must be non-negative")

    aligned = _AlignedArrays(returns, features)
    coeffs, intercept, covariance = _ridge_regression(
        aligned.returns, aligned.exposures, regularisation=regularisation
    )

    predictions = aligned.exposures @ coeffs + intercept
    residuals = aligned.returns - predictions
    total_return = float(np.sum(aligned.returns))
    average_return = float(np.mean(aligned.returns))
    total_volatility = float(np.std(aligned.returns, ddof=1 if len(aligned.returns) > 1 else 0))
    sharpe_ratio: float | None
    if total_volatility > 0:
        sharpe_ratio = average_return / total_volatility * np.sqrt(252.0)
    else:
        sharpe_ratio = None

    ss_total = float(np.sum((aligned.returns - average_return) ** 2))
    ss_residual = float(np.sum(residuals**2))
    r_squared = 0.0 if ss_total == 0 else 1.0 - ss_residual / ss_total
    residual_mean = float(np.mean(residuals))

    contributions: list[FeatureContribution] = []
    for idx, name in enumerate(aligned.names):
        exposure = aligned.exposures[:, idx]
        mean_exposure = float(np.mean(exposure))
        contribution = float(coeffs[idx] * mean_exposure)
        variance = float(covariance[idx, idx])
        if variance > 0:
            t_stat = float(coeffs[idx] / np.sqrt(variance))
        else:
            t_stat = None
        contributions.append(
            FeatureContribution(
                name=name,
                coefficient=float(coeffs[idx]),
                mean_exposure=mean_exposure,
                contribution=contribution,
                t_stat=t_stat,
            )
        )

    return AttributionResult(
        total_return=total_return,
        average_return=average_return,
        total_volatility=total_volatility,
        sharpe_ratio=sharpe_ratio,
        residual_mean=residual_mean,
        r_squared=float(max(min(r_squared, 1.0), -1.0)),
        intercept=float(intercept),
        contributions=tuple(contributions),
    )


def result_to_dataframe(result: AttributionResult) -> "np.ndarray":
    """Return a numpy structured array suited for DataFrame construction."""

    dtype = [
        ("name", "U64"),
        ("coefficient", "f8"),
        ("mean_exposure", "f8"),
        ("contribution", "f8"),
        ("t_stat", "f8"),
    ]
    rows = []
    for contrib in result.contributions:
        rows.append(
            (
                contrib.name,
                contrib.coefficient,
                contrib.mean_exposure,
                contrib.contribution,
                contrib.t_stat if contrib.t_stat is not None else np.nan,
            )
        )
    return np.array(rows, dtype=dtype)
