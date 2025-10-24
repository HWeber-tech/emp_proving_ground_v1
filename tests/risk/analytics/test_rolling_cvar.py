from __future__ import annotations

import math

import numpy as np
import pytest

from src.risk.analytics.expected_shortfall import compute_historical_expected_shortfall
from src.risk.analytics.rolling_cvar import RollingCVaRMeasurement, RollingCVaRMonitor


def test_monitor_requires_valid_configuration() -> None:
    with pytest.raises(ValueError):
        RollingCVaRMonitor(window=0)
    with pytest.raises(ValueError):
        RollingCVaRMonitor(window=10, confidence=1.5)
    with pytest.raises(ValueError):
        RollingCVaRMonitor(window=5, min_periods=0)
    with pytest.raises(ValueError):
        RollingCVaRMonitor(window=5, min_periods=6)


def test_monitor_computes_cvar_over_window() -> None:
    monitor = RollingCVaRMonitor(window=4, confidence=0.95)
    series = [0.01, -0.02, -0.015, 0.005, -0.03]
    expected_windows = [series[:4], series[1:]]

    results: list[RollingCVaRMeasurement | None] = []
    for value in series:
        results.append(monitor.observe(value))

    assert results[:3] == [None, None, None]
    first = results[3]
    second = results[4]
    assert isinstance(first, RollingCVaRMeasurement)
    assert isinstance(second, RollingCVaRMeasurement)

    for measurement, window in zip((first, second), expected_windows, strict=True):
        arr = np.asarray(window, dtype=float)
        es = compute_historical_expected_shortfall(arr, confidence=0.95)
        percentile = (1.0 - 0.95) * 100.0
        var_threshold = float(np.percentile(arr, percentile))
        breaches = int(np.sum(arr <= var_threshold))

        assert math.isclose(measurement.cvar, es.value, rel_tol=1e-9)
        assert math.isclose(measurement.var, max(-var_threshold, 0.0), rel_tol=1e-9)
        assert math.isclose(measurement.mean, float(np.mean(arr)), rel_tol=1e-9)
        assert math.isclose(
            measurement.std,
            float(np.std(arr, ddof=1)),
            rel_tol=1e-9,
        )
        assert measurement.breaches == breaches
        assert measurement.sample_size == len(window)
        assert measurement.confidence == pytest.approx(0.95)
        assert measurement.window == 4


def test_monitor_ignores_non_finite_observations() -> None:
    monitor = RollingCVaRMonitor(window=3, confidence=0.9, min_periods=2)
    assert monitor.observe(-0.01) is None
    assert monitor.observe(float("nan")) is None
    assert monitor.observe(float("inf")) is None
    result = monitor.observe(-0.05)
    assert isinstance(result, RollingCVaRMeasurement)
    assert len(monitor) == 2
    assert monitor.values() == (-0.01, -0.05)

    monitor.extend([0.02, float("nan"), -0.03])
    assert len(monitor) == 3
    assert isinstance(monitor.current, RollingCVaRMeasurement)
