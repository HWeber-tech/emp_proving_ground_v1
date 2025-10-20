from __future__ import annotations

import pytest

from src.trading.execution.execution_model import calculate_dimensionless_delta_hat


def test_dimensionless_delta_hat_uses_spread_floor_when_wider_than_observed() -> None:
    delta_hat = calculate_dimensionless_delta_hat(
        mid_now=100.0,
        mid_future=100.03,
        tick_size=0.01,
        spread_ticks=1.0,
        spread_floor_ticks=2.0,
    )

    expected = 0.03 / (0.01 * 2.0)
    assert delta_hat == pytest.approx(expected)


def test_dimensionless_delta_hat_prefers_observed_spread_when_larger() -> None:
    delta_hat = calculate_dimensionless_delta_hat(
        mid_now=100.0,
        mid_future=99.985,
        tick_size=0.01,
        spread_ticks=3.0,
        spread_floor_ticks=2.0,
    )

    expected = (99.985 - 100.0) / (0.01 * 3.0)
    assert delta_hat == pytest.approx(expected)


def test_dimensionless_delta_hat_returns_zero_when_inputs_invalid() -> None:
    assert calculate_dimensionless_delta_hat(None, 100.0, 0.01, 1.0) == 0.0
    assert calculate_dimensionless_delta_hat(100.0, None, 0.01, 1.0) == 0.0
    assert calculate_dimensionless_delta_hat(100.0, 100.01, 0.0, 1.0) == 0.0
    assert (
        calculate_dimensionless_delta_hat(
            100.0,
            100.01,
            0.01,
            0.0,
            spread_floor_ticks=0.0,
        )
        == 0.0
    )
