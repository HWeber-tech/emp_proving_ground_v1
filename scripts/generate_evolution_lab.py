"""CLI for generating the Evolution Lab leaderboard document."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.evolution.experiments import (
    GARunConfig,
    render_evolution_lab_markdown,
    run_ma_crossover_lab,
)


def _make_synthetic_prices(length: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(loc=0.15, scale=0.6, size=length))
    prices = 100 + base
    return pd.Series(prices)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length",
        type=int,
        default=256,
        help="Number of price points to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for synthetic price generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/evolution_lab.md"),
        help="Path where the Markdown report should be written",
    )
    parser.add_argument(
        "--experiment",
        default="ma_crossover_ga",
        help="Experiment name to record in the manifest",
    )
    parser.add_argument(
        "--dataset-name",
        default="synthetic_trend_v1",
        help="Dataset identifier included in the manifest",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the champion genome in the strategy registry",
    )
    parser.add_argument(
        "--registry-db",
        type=Path,
        default=None,
        help="Optional path to the registry database",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=8,
        help="Number of GA generations",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=18,
        help="Population size for the GA",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=3,
        help="Number of elites to preserve each generation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prices = _make_synthetic_prices(args.length, args.seed)
    config = GARunConfig(
        population_size=args.population,
        generations=args.generations,
        elite_count=args.elite_count,
        seed=args.seed,
    )
    report = run_ma_crossover_lab(
        prices,
        experiment_name=args.experiment,
        dataset_name=args.dataset_name,
        config=config,
        register_champion=args.register,
        registry_db_path=args.registry_db,
        notes={"data": "synthetic"},
    )
    markdown = render_evolution_lab_markdown(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
