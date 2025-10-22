"""Monte Carlo simulation utilities for price path analysis.

The README roadmap calls for a reusable Monte Carlo simulation framework.  This
module provides a focused implementation centred on geometric Brownian motion
paths, which aligns with the rest of the risk analytics stack.  The
``simulate_geometric_brownian_motion`` helper returns a
``MonteCarloSimulationResult`` object with convenience accessors for common risk
statistics so downstream components can consume simulated distributions without
reinventing boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["MonteCarloSimulationResult", "simulate_geometric_brownian_motion"]


_EPSILON: Final[float] = 1e-12


@dataclass(slots=True)
class MonteCarloSimulationResult:
    """Container exposing metadata and summary helpers for Monte Carlo paths."""

    paths: np.ndarray
    time_grid: np.ndarray
    drift: float
    volatility: float
    dt: float

    def __post_init__(self) -> None:
        if self.paths.ndim != 2:
            raise ValueError("paths must be a 2D array of shape (simulations, steps+1)")
        if self.time_grid.ndim != 1:
            raise ValueError("time_grid must be one-dimensional")
        if self.paths.shape[1] != self.time_grid.shape[0]:
            raise ValueError("time_grid length must equal number of path steps")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.volatility < 0.0:
            raise ValueError("volatility cannot be negative")
        if np.any(self.paths[:, 0] <= 0.0):
            raise ValueError("starting prices must be positive")

    @property
    def simulations(self) -> int:
        """Number of Monte Carlo paths."""

        return int(self.paths.shape[0])

    @property
    def steps(self) -> int:
        """Number of discrete steps in each path."""

        return int(self.paths.shape[1] - 1)

    @property
    def start_price(self) -> float:
        """Initial price shared by all simulated paths."""

        return float(self.paths[0, 0])

    def terminal_prices(self) -> np.ndarray:
        """Return a copy of the terminal prices for each simulated path."""

        return np.array(self.paths[:, -1], copy=True)

    def log_returns(self) -> np.ndarray:
        """Matrix of log returns for each path across time steps."""

        return np.diff(np.log(self.paths), axis=1)

    def losses(self) -> np.ndarray:
        """Compute terminal losses relative to the starting price."""

        terminal = self.terminal_prices()
        start = self.paths[:, 0]
        return start - terminal

    def value_at_risk(self, confidence: float = 0.99) -> float:
        """Return the loss threshold that is not exceeded with ``confidence``.

        The value is expressed as a positive loss amount.
        """

        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        losses = self.losses()
        quantile = float(np.quantile(losses, confidence))
        return max(quantile, 0.0)

    def expected_shortfall(self, confidence: float = 0.99) -> float:
        """Return the Conditional VaR (Expected Shortfall) for the loss tail."""

        var = self.value_at_risk(confidence)
        losses = self.losses()
        tail = losses[losses >= var - _EPSILON]
        if tail.size == 0:
            return var
        return float(np.mean(tail))

    def summary(self, confidence: float = 0.99) -> dict[str, float]:
        """Summarise key statistics for reporting pipelines."""

        terminal = self.terminal_prices()
        return {
            "simulations": float(self.simulations),
            "steps": float(self.steps),
            "start_price": self.start_price,
            "mean_terminal_price": float(np.mean(terminal)),
            "std_terminal_price": float(np.std(terminal, ddof=0)),
            "value_at_risk": self.value_at_risk(confidence),
            "expected_shortfall": self.expected_shortfall(confidence),
        }


def simulate_geometric_brownian_motion(
    start_price: float,
    *,
    drift: float,
    volatility: float,
    horizon: float,
    steps: int,
    simulations: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> MonteCarloSimulationResult:
    """Simulate price paths under geometric Brownian motion dynamics.

    Parameters
    ----------
    start_price:
        Initial asset price (must be strictly positive).
    drift:
        Continuous compounding drift of the process.
    volatility:
        Annualised volatility (non-negative).
    horizon:
        Total time horizon expressed in years.
    steps:
        Number of discrete time steps used to span the ``horizon``.
    simulations:
        Number of Monte Carlo scenarios to simulate.
    seed:
        Optional random seed for deterministic reproducibility.
    antithetic:
        When ``True`` an antithetic variates scheme is used to reduce variance.
    """

    if start_price <= 0.0:
        raise ValueError("start_price must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if volatility < 0.0:
        raise ValueError("volatility cannot be negative")

    dt = horizon / steps
    time_grid = np.linspace(0.0, horizon, steps + 1, dtype=float)
    rng = np.random.default_rng(seed)

    if antithetic:
        half = (simulations + 1) // 2
        base = rng.standard_normal((half, steps))
        normals = np.empty((half * 2, steps), dtype=float)
        normals[0::2] = base
        normals[1::2] = -base
        normals = normals[:simulations]
    else:
        normals = rng.standard_normal((simulations, steps))

    drift_term = (drift - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt) * normals
    log_increments = drift_term + diffusion
    cumulative = np.cumsum(log_increments, axis=1)

    paths = np.empty((simulations, steps + 1), dtype=float)
    paths[:, 0] = start_price
    paths[:, 1:] = start_price * np.exp(cumulative)

    return MonteCarloSimulationResult(
        paths=paths,
        time_grid=time_grid,
        drift=drift,
        volatility=volatility,
        dt=dt,
    )

