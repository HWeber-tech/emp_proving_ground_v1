"""Rolling CVaR monitor supporting the high-impact roadmap."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Sequence

import math

import numpy as np

from .expected_shortfall import compute_historical_expected_shortfall

__all__ = [
    "RollingCVaRMeasurement",
    "RollingCVaRMonitor",
]


@dataclass(slots=True)
class RollingCVaRMeasurement:
    """Summary of the current Conditional Value at Risk estimate."""

    cvar: float
    var: float
    mean: float
    std: float
    breaches: int
    confidence: float
    sample_size: int
    window: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable representation of the measurement."""

        return {
            "cvar": self.cvar,
            "var": self.var,
            "mean": self.mean,
            "std": self.std,
            "breaches": self.breaches,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "window": self.window,
        }


class RollingCVaRMonitor:
    """Track rolling Conditional Value at Risk for a returns stream."""

    def __init__(
        self,
        *,
        window: int,
        confidence: float = 0.99,
        min_periods: int | None = None,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if min_periods is None:
            min_periods = window
        if min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if min_periods > window:
            raise ValueError("min_periods cannot exceed window")

        self._window = int(window)
        self._confidence = float(confidence)
        self._min_periods = int(min_periods)
        self._values: Deque[float] = deque(maxlen=self._window)
        self._current: RollingCVaRMeasurement | None = None

    @property
    def window(self) -> int:
        """Configured rolling window size."""

        return self._window

    @property
    def confidence(self) -> float:
        """Confidence level used for the CVaR estimate."""

        return self._confidence

    @property
    def min_periods(self) -> int:
        """Minimum number of observations required before emitting estimates."""

        return self._min_periods

    @property
    def current(self) -> RollingCVaRMeasurement | None:
        """Return the last computed measurement, if available."""

        return self._current

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._values)

    def __iter__(self) -> Iterator[float]:  # pragma: no cover - simple delegation
        return iter(self._values)

    def observe(self, value: float | int | np.floating) -> RollingCVaRMeasurement | None:
        """Record a new return observation and update the CVaR estimate."""

        numeric = _coerce_float(value)
        if numeric is None:
            return self._current

        self._values.append(numeric)
        if len(self._values) < self._min_periods:
            self._current = None
            return None

        self._current = self._compute_measurement()
        return self._current

    def extend(self, values: Sequence[float] | Iterable[float]) -> RollingCVaRMeasurement | None:
        """Ingest an iterable of observations, returning the latest measurement."""

        result: RollingCVaRMeasurement | None = None
        for value in values:
            result = self.observe(value)
        return result

    def reset(self) -> None:
        """Clear the rolling window and any computed state."""

        self._values.clear()
        self._current = None

    def values(self) -> tuple[float, ...]:
        """Return the current rolling window values."""

        return tuple(self._values)

    def _compute_measurement(self) -> RollingCVaRMeasurement:
        window = np.asarray(self.values(), dtype=float)
        es_result = compute_historical_expected_shortfall(
            window,
            confidence=self._confidence,
        )
        percentile = (1.0 - self._confidence) * 100.0
        var_threshold = float(np.percentile(window, percentile))
        tail = window[window <= var_threshold]
        breaches = int(tail.size)
        ddof = 1 if window.size > 1 else 0
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=ddof))
        return RollingCVaRMeasurement(
            cvar=es_result.value,
            var=float(max(-var_threshold, 0.0)),
            mean=mean,
            std=std,
            breaches=breaches,
            confidence=self._confidence,
            sample_size=int(window.size),
            window=self._window,
        )


def _coerce_float(value: object) -> float | None:
    """Best-effort conversion that rejects non-finite values."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric
