"""Expected Shortfall estimators supporting the high-impact roadmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "ExpectedShortfallResult",
    "compute_historical_expected_shortfall",
    "compute_parametric_expected_shortfall",
]


@dataclass(slots=True)
class ExpectedShortfallResult:
    """Container summarising an Expected Shortfall calculation."""

    value: float
    confidence: float
    sample_size: int

    def as_dict(self) -> dict[str, float]:
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
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("returns must contain at least one finite element")
    return array


def compute_historical_expected_shortfall(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
) -> ExpectedShortfallResult:
    """Compute Expected Shortfall using the empirical loss tail."""

    confidence = _normalise_confidence(confidence)
    sample = _prepare_returns(returns)
    percentile = (1.0 - confidence) * 100.0
    var_threshold = np.percentile(sample, percentile)
    tail = sample[sample <= var_threshold]
    if tail.size == 0:
        es_value = float(max(-var_threshold, 0.0))
    else:
        es_value = float(max(-float(np.mean(tail)), 0.0))
    return ExpectedShortfallResult(
        value=es_value,
        confidence=confidence,
        sample_size=int(sample.size),
    )


def compute_parametric_expected_shortfall(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
    ddof: int = 1,
) -> ExpectedShortfallResult:
    """Compute Expected Shortfall assuming a normal distribution."""

    confidence = _normalise_confidence(confidence)
    sample = _prepare_returns(returns)
    mean = float(np.mean(sample))
    std = float(np.std(sample, ddof=ddof if sample.size > ddof else 0))
    if std <= 0.0:
        es_value = float(max(-mean, 0.0))
    else:
        # Closed-form ES for normal distribution
        from statistics import NormalDist

        dist = NormalDist(mu=mean, sigma=std)
        var_threshold = dist.inv_cdf(1.0 - confidence)
        pdf = dist.pdf(var_threshold)
        es = mean - std * pdf / (1.0 - confidence)
        es_value = float(max(-es, 0.0))
    return ExpectedShortfallResult(
        value=es_value,
        confidence=confidence,
        sample_size=int(sample.size),
    )
