"""Value-at-Risk estimators aligned with the high-impact roadmap.

The roadmap calls for historical, parametric, and Monte Carlo VaR calculations
that can be plugged into pre-trade risk checks.  These helpers keep the
implementation small and dependency-light while ensuring consistent sanitisation
and statistical assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "VarResult",
    "compute_historical_var",
    "compute_parametric_var",
    "compute_monte_carlo_var",
]


@dataclass(slots=True)
class VarResult:
    """Container summarising a VaR calculation."""

    value: float
    confidence: float
    sample_size: int

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable representation."""

        return {
            "value": self.value,
            "confidence": self.confidence,
            "sample_size": float(self.sample_size),
        }


def _normalise_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def _prepare_returns(returns: Sequence[float] | Iterable[float]) -> np.ndarray:
    array = np.asarray(list(returns), dtype=float)
    if array.size == 0:
        raise ValueError("returns must contain at least one element")
    # Drop NaNs/Infs defensively
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("returns must contain at least one finite element")
    return array


def compute_historical_var(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
) -> VarResult:
    """Compute historical Value-at-Risk via empirical quantiles."""

    confidence = _normalise_confidence(confidence)
    sample = _prepare_returns(returns)
    percentile = (1.0 - confidence) * 100.0
    threshold = np.percentile(sample, percentile)
    value = float(max(-threshold, 0.0))
    return VarResult(value=value, confidence=confidence, sample_size=int(sample.size))


def compute_parametric_var(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
    ddof: int = 1,
) -> VarResult:
    """Compute parametric VaR assuming normally distributed returns."""

    confidence = _normalise_confidence(confidence)
    sample = _prepare_returns(returns)
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=ddof if sample.size > ddof else 0))
    if std <= 0.0:
        threshold = mean
    else:
        z_score = NormalDist().inv_cdf(1.0 - confidence)
        threshold = mean + std * z_score
    value = float(max(-threshold, 0.0))
    return VarResult(value=value, confidence=confidence, sample_size=int(sample.size))


def compute_monte_carlo_var(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
    simulations: int = 10000,
    seed: int | None = None,
) -> VarResult:
    """Estimate VaR using Monte Carlo sampling from a normal approximation."""

    if simulations <= 0:
        raise ValueError("simulations must be positive")
    confidence = _normalise_confidence(confidence)
    sample = _prepare_returns(returns)
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=1 if sample.size > 1 else 0))

    if std <= 0.0:
        simulated = np.repeat(mean, simulations)
    else:
        rng = np.random.default_rng(seed)
        simulated = rng.normal(mean, std, size=simulations)

    percentile = (1.0 - confidence) * 100.0
    threshold = float(np.percentile(simulated, percentile))
    value = float(max(-threshold, 0.0))
    return VarResult(value=value, confidence=confidence, sample_size=int(sample.size))
