from __future__ import annotations

import math

import numpy as np

from src.risk.analytics.monte_carlo import (
    MonteCarloSimulationResult,
    simulate_geometric_brownian_motion,
)


def test_simulate_gbm_shapes_and_metadata() -> None:
    result = simulate_geometric_brownian_motion(
        100.0,
        drift=0.05,
        volatility=0.2,
        horizon=1.0,
        steps=252,
        simulations=128,
        seed=42,
    )

    assert isinstance(result, MonteCarloSimulationResult)
    assert result.paths.shape == (128, 253)
    assert result.time_grid.shape == (253,)
    assert math.isclose(result.time_grid[0], 0.0)
    assert math.isclose(result.time_grid[-1], 1.0)
    assert result.simulations == 128
    assert result.steps == 252
    assert result.start_price == 100.0


def test_simulate_gbm_zero_volatility_matches_closed_form() -> None:
    drift = 0.08
    result = simulate_geometric_brownian_motion(
        50.0,
        drift=drift,
        volatility=0.0,
        horizon=0.5,
        steps=10,
        simulations=3,
        seed=1,
    )

    expected = 50.0 * np.exp(drift * result.time_grid)
    np.testing.assert_allclose(result.paths, np.tile(expected, (3, 1)), rtol=1e-12)


def test_antithetic_variates_produce_opposing_diffusion() -> None:
    simulations = 10
    steps = 5
    result = simulate_geometric_brownian_motion(
        100.0,
        drift=0.03,
        volatility=0.25,
        horizon=1.0,
        steps=steps,
        simulations=simulations,
        seed=123,
        antithetic=True,
    )

    log_returns = result.log_returns()
    drift_term = (result.drift - 0.5 * result.volatility**2) * result.dt
    diffusion = log_returns - drift_term

    # Pair up the diffusion component for antithetic samples.  The final sample
    # may be unpaired when ``simulations`` is odd, so slice to the nearest even
    # count to assert symmetry.
    even_count = diffusion.shape[0] - diffusion.shape[0] % 2
    paired = diffusion[:even_count].reshape(-1, 2, steps)
    np.testing.assert_allclose(paired[:, 0, :], -paired[:, 1, :], atol=1e-8)


def test_summary_and_risk_metrics() -> None:
    result = simulate_geometric_brownian_motion(
        200.0,
        drift=-0.1,
        volatility=0.05,
        horizon=1.0,
        steps=64,
        simulations=5000,
        seed=7,
    )

    var = result.value_at_risk(0.95)
    es = result.expected_shortfall(0.95)
    assert var >= 0.0
    assert es >= var

    summary = result.summary(0.95)
    assert math.isclose(summary["value_at_risk"], var, rel_tol=1e-9)
    assert math.isclose(summary["expected_shortfall"], es, rel_tol=1e-9)
    assert summary["simulations"] == float(result.simulations)
    assert summary["steps"] == float(result.steps)
