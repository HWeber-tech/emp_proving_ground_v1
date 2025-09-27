"""Unit tests for the high-impact risk analytics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.analytics import (
    compute_historical_expected_shortfall,
    compute_historical_var,
    compute_monte_carlo_var,
    compute_parametric_expected_shortfall,
    compute_parametric_var,
)


def test_historical_var_matches_percentile() -> None:
    returns = np.array([-0.04, -0.02, -0.01, 0.0, 0.01, 0.02])
    expected_threshold = float(np.percentile(returns, 5.0))

    result = compute_historical_var(returns, confidence=0.95)

    assert result.confidence == pytest.approx(0.95)
    assert result.value == pytest.approx(max(-expected_threshold, 0.0))
    assert result.sample_size == len(returns)


def test_parametric_var_uses_normal_approximation() -> None:
    returns = np.array([-0.03, -0.015, 0.005, 0.012, -0.02, 0.01])
    result = compute_parametric_var(returns, confidence=0.975)

    assert result.confidence == pytest.approx(0.975)
    assert result.value >= 0


def test_monte_carlo_var_is_reproducible_with_seed() -> None:
    returns = np.array([-0.015, 0.003, -0.02, 0.007, -0.005, 0.002])

    first = compute_monte_carlo_var(
        returns, confidence=0.99, simulations=1_000, seed=42
    )
    second = compute_monte_carlo_var(
        returns, confidence=0.99, simulations=1_000, seed=42
    )

    assert first.value == pytest.approx(second.value)
    assert first.sample_size == len(returns)
    assert second.sample_size == len(returns)


def test_expected_shortfall_not_less_than_var() -> None:
    returns = np.array([-0.025, -0.018, 0.004, -0.01, 0.006, -0.022])
    var_result = compute_historical_var(returns, confidence=0.95)
    es_result = compute_historical_expected_shortfall(returns, confidence=0.95)

    assert es_result.value >= var_result.value
    assert es_result.sample_size == len(returns)


def test_parametric_expected_shortfall_handles_zero_volatility() -> None:
    returns = np.array([-0.01, -0.01, -0.01, -0.01])
    result = compute_parametric_expected_shortfall(returns, confidence=0.99)

    assert result.value == pytest.approx(0.01)
    assert result.sample_size == len(returns)
