from __future__ import annotations

import numpy as np
import pandas as pd

from src.evolution.experiments.ma_crossover_ga import GARunConfig, run_ga_experiment
from src.evolution.experiments.reporting import render_leaderboard_markdown


def _synthetic_prices(seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(loc=0.1, scale=0.3, size=96))
    return pd.Series(100.0 + base, name="price")


def test_ga_manifest_contains_replay_metadata() -> None:
    prices = _synthetic_prices()
    config = GARunConfig(population_size=6, generations=4, elite_count=2, seed=7)

    result = run_ga_experiment(prices, config)
    manifest = result.to_manifest(
        experiment="ma_crossover_ga",
        config=config,
        dataset_name="unit_test_series",
        dataset_metadata={"length": len(prices)},
        code_version="test-hash",
        notes="unit-test",
        replay_command="python scripts/generate_evolution_lab.py --seed 7",
    )

    assert manifest["experiment"] == "ma_crossover_ga"
    assert manifest["seed"] == 7
    assert manifest["config"]["population_size"] == config.population_size
    assert manifest["dataset"]["name"] == "unit_test_series"
    assert manifest["dataset"]["metadata"]["length"] == len(prices)
    assert manifest["best_genome"]["long_window"] > manifest["best_genome"]["short_window"]
    assert len(manifest["leaderboard"]) == config.generations
    assert manifest["leaderboard"][0]["generation"] == 1
    assert manifest["replay"]["command"].startswith("python")
    assert manifest["code_version"] == "test-hash"
    assert manifest["notes"] == "unit-test"


def test_render_leaderboard_markdown_sorts_by_fitness() -> None:
    base_manifest = {
        "experiment": "ma_crossover_ga",
        "seed": 10,
        "best_metrics": {
            "fitness": 1.2,
            "sharpe": 1.8,
            "sortino": 2.2,
            "max_drawdown": 0.12,
            "total_return": 0.34,
        },
        "best_genome": {
            "short_window": 8,
            "long_window": 42,
            "risk_fraction": 0.2,
            "use_var_guard": True,
            "use_drawdown_guard": True,
        },
    }
    weaker_manifest = {
        "experiment": "control",
        "seed": 11,
        "best_metrics": {
            "fitness": -0.3,
            "sharpe": 0.1,
            "sortino": 0.2,
            "max_drawdown": 0.4,
            "total_return": -0.05,
        },
        "best_genome": {
            "short_window": 5,
            "long_window": 25,
            "risk_fraction": 0.1,
            "use_var_guard": False,
            "use_drawdown_guard": False,
        },
    }

    markdown = render_leaderboard_markdown([weaker_manifest, base_manifest])

    # Ensure header and ordering
    assert markdown.startswith("| Experiment | Seed |")
    rows = [line for line in markdown.splitlines() if line.startswith("|")][2:]
    assert rows[0].startswith("| ma_crossover_ga | 10 | 1.200")
    assert "control" in rows[1]


def test_render_leaderboard_markdown_handles_empty() -> None:
    assert render_leaderboard_markdown([]) == "No experiments recorded yet.\n"
