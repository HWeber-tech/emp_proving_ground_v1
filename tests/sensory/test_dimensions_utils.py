import pytest

from src.sensory.organs.dimensions.utils import calculate_divergence


def test_calculate_divergence_handles_zero_baseline() -> None:
    price_series = [100.0, 101.0]
    indicator_series = [0.0, 0.0]

    result = calculate_divergence(price_series, indicator_series)

    assert result == pytest.approx(0.005)
