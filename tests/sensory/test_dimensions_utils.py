import pytest

from src.sensory.organs.dimensions.utils import calculate_divergence, calculate_volatility


def test_calculate_divergence_handles_zero_baseline() -> None:
    price_series = [100.0, 101.0]
    indicator_series = [0.0, 0.0]

    result = calculate_divergence(price_series, indicator_series)

    assert result == pytest.approx(0.005)


def test_calculate_volatility_returns_fallback_for_insufficient_points() -> None:
    result = calculate_volatility([100.0], period=1)

    assert result == 0.5
