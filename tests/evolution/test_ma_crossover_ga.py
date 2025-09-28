from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.evolution.experiments.ma_crossover_ga import (
    GARunConfig,
    MovingAverageGenome,
    MovingAverageGenomeBounds,
    evaluate_genome_fitness,
    run_ga_experiment,
)


def _make_trending_prices(length: int = 200) -> pd.Series:
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(loc=0.2, scale=0.5, size=length))
    prices = 100 + base.clip(min=-80)
    return pd.Series(prices)


def test_bounds_clamp_enforces_gap_and_ranges() -> None:
    bounds = MovingAverageGenomeBounds(short_window=(3, 10), long_window=(15, 40), risk_fraction=(0.1, 0.3), min_window_gap=4)
    genome = MovingAverageGenome(short_window=20, long_window=22, risk_fraction=0.9, use_var_guard=False, use_drawdown_guard=True)

    clamped = bounds.clamp(genome)

    assert bounds.short_window[0] <= clamped.short_window <= bounds.short_window[1]
    assert bounds.long_window[0] <= clamped.long_window <= bounds.long_window[1]
    assert clamped.long_window - clamped.short_window >= bounds.min_window_gap
    assert bounds.risk_fraction[0] <= clamped.risk_fraction <= bounds.risk_fraction[1]
    assert clamped.use_var_guard is False
    assert clamped.use_drawdown_guard is True


def test_evaluate_genome_fitness_returns_positive_score_for_trend() -> None:
    prices = _make_trending_prices()
    genome = MovingAverageGenome(short_window=5, long_window=25, risk_fraction=0.25)

    metrics = evaluate_genome_fitness(prices, genome)

    assert metrics.total_return > 0
    assert math.isfinite(metrics.sharpe)
    assert math.isfinite(metrics.sortino)
    assert metrics.max_drawdown >= 0


def test_ga_experiment_improves_population() -> None:
    prices = _make_trending_prices(160)
    config = GARunConfig(population_size=10, generations=6, elite_count=2, seed=123)

    result = run_ga_experiment(prices, config)

    assert isinstance(result.best_genome, MovingAverageGenome)
    assert result.best_metrics.fitness == max(m.fitness for _, m in result.population_history)
    assert result.best_genome.long_window > result.best_genome.short_window
    assert len(result.population_history) == config.generations


def test_ga_result_to_frame_exposes_generation_history() -> None:
    prices = _make_trending_prices(120)
    config = GARunConfig(population_size=8, generations=4, elite_count=2, seed=7)

    result = run_ga_experiment(prices, config)
    frame = result.to_frame()

    assert list(frame.columns) == [
        "generation",
        "short_window",
        "long_window",
        "risk_fraction",
        "use_var_guard",
        "use_drawdown_guard",
        "fitness",
        "sharpe",
        "sortino",
        "max_drawdown",
        "total_return",
    ]
    assert frame.shape[0] == config.generations
