from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["AnomalyEvaluation", "BasicAnomalyDetector"]


@dataclass(slots=True)
class AnomalyEvaluation:
    """Simple summary of anomaly statistics for a numeric sample."""

    sample_size: int
    mean: float
    std_dev: float
    latest: float
    z_score: float
    is_anomaly: bool


class BasicAnomalyDetector:
    """Lightweight z-score detector used by the ANOMALY organ."""

    def __init__(
        self,
        *,
        window: int = 32,
        min_samples: int = 8,
        z_threshold: float = 3.0,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if min_samples <= 1:
            raise ValueError("min_samples must be greater than 1")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")

        self._window = int(window)
        self._min_samples = int(min_samples)
        self._z_threshold = float(z_threshold)

    def evaluate(self, data: Sequence[float] | Iterable[float] | pd.Series) -> AnomalyEvaluation:
        """Compute anomaly statistics using a rolling z-score."""

        values = self._normalise_values(data)
        if not values:
            return AnomalyEvaluation(0, 0.0, 0.0, 0.0, 0.0, False)

        windowed = values[-self._window :]
        sample_size = len(windowed)
        mean = fmean(windowed)
        std_dev = pstdev(windowed) if sample_size > 1 else 0.0
        latest = windowed[-1]
        if std_dev <= 0.0:
            z_score = 0.0
        else:
            z_score = (latest - mean) / std_dev

        is_anomaly = sample_size >= self._min_samples and abs(z_score) >= self._z_threshold
        return AnomalyEvaluation(sample_size, float(mean), float(std_dev), float(latest), float(z_score), is_anomaly)

    def _normalise_values(
        self, data: Sequence[float] | Iterable[float] | pd.Series
    ) -> list[float]:
        if isinstance(data, pd.Series):
            iterable = data.tolist()
        else:
            iterable = list(data)

        cleaned: list[float] = []
        for raw in iterable:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not isfinite(value):
                continue
            cleaned.append(value)
        return cleaned

    @property
    def window(self) -> int:
        return self._window

    @property
    def min_samples(self) -> int:
        return self._min_samples

    @property
    def z_threshold(self) -> float:
        return self._z_threshold
