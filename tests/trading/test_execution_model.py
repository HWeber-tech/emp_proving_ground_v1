from __future__ import annotations

import pytest

from src.trading.execution.execution_model import (
    calculate_dimensionless_delta_hat,
    calculate_dual_horizon_delta_hats,
)


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


def test_calculate_dual_horizon_delta_hats_returns_canonical_labels() -> None:
    result = calculate_dual_horizon_delta_hats(
        mid_now=100.0,
        tick_size=0.01,
        event_mid_prices={
            1: 100.05,
            "EV5": 100.02,
            "20": 99.98,
            "invalid": "n/a",
        },
        wall_time_mid_prices={
            "100ms": 100.01,
            "0.5s": 100.02,
            "2.00S": 100.3,
        },
        event_spread_ticks=1.5,
        wall_time_spread_ticks={"500ms": 1.1, "100ms": 1.0, "2s": 1.2},
        event_spread_floor_ticks={"ev1": 1.0},
        wall_time_spread_floor_ticks=0.8,
    )

    assert tuple(result["event"].keys()) == ("ev1", "ev5", "ev20")
    assert tuple(result["time"].keys()) == ("100ms", "500ms", "2s")

    expected_ev1 = calculate_dimensionless_delta_hat(
        100.0,
        100.05,
        0.01,
        1.5,
        spread_floor_ticks=1.0,
    )
    expected_ev5 = calculate_dimensionless_delta_hat(100.0, 100.02, 0.01, 1.5)
    expected_ev20 = calculate_dimensionless_delta_hat(100.0, 99.98, 0.01, 1.5)

    expected_100ms = calculate_dimensionless_delta_hat(
        100.0,
        100.01,
        0.01,
        1.0,
        spread_floor_ticks=0.8,
    )
    expected_500ms = calculate_dimensionless_delta_hat(
        100.0,
        100.02,
        0.01,
        1.1,
        spread_floor_ticks=0.8,
    )
    expected_2s = calculate_dimensionless_delta_hat(
        100.0,
        100.3,
        0.01,
        1.2,
        spread_floor_ticks=0.8,
    )

    assert result["event"]["ev1"] == pytest.approx(expected_ev1)
    assert result["event"]["ev5"] == pytest.approx(expected_ev5)
    assert result["event"]["ev20"] == pytest.approx(expected_ev20)
    assert result["time"]["100ms"] == pytest.approx(expected_100ms)
    assert result["time"]["500ms"] == pytest.approx(expected_500ms)
    assert result["time"]["2s"] == pytest.approx(expected_2s)


def test_calculate_dual_horizon_delta_hats_handles_invalid_inputs() -> None:
    result = calculate_dual_horizon_delta_hats(
        mid_now=None,
        tick_size=0.01,
        event_mid_prices={1: 100.1},
        wall_time_mid_prices={"100ms": 100.05},
        event_spread_ticks=1.0,
        wall_time_spread_ticks=1.0,
    )

    assert result == {"event": {}, "time": {}}

    result = calculate_dual_horizon_delta_hats(
        mid_now=100.0,
        tick_size=0.01,
        event_mid_prices={"mystery": 101.0},
        wall_time_mid_prices={"mystery": 101.0},
        event_spread_ticks=1.0,
        wall_time_spread_ticks=1.0,
    )

    assert result == {"event": {}, "time": {}}
