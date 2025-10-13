"""Offline genetic algorithm experiments for moving-average crossover tuning.

This module operationalises the Phase 2B roadmap requirement by providing a
minimal-yet-practical genome schema together with a small GA loop that can be
executed offline.  The focus is on:

* A canonical genome representation for the baseline moving-average strategy
  augmented with risk toggles.
* Fitness evaluation driven by Sharpe, Sortino, max drawdown, and total return.
* Safe crossover and mutation operators with guardrails to avoid invalid
  configurations.
* A lightweight orchestration helper that produces reproducible experiment
  results which downstream orchestration can surface in dashboards or docs.

The implementation intentionally avoids tying into the live trading loop so it
can run cheaply during CI or analysis notebooks, fulfilling the roadmap's
requirement for reproducible GA experiments.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.genome.models.genome import DecisionGenome, new_genome
from src.trading.monitoring.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

__all__ = [
    "MovingAverageGenome",
    "MovingAverageGenomeBounds",
    "FitnessWeights",
    "FitnessMetrics",
    "GARunConfig",
    "GARunResult",
    "evaluate_genome_fitness",
    "run_ga_experiment",
]


@dataclass(slots=True)
class MovingAverageGenome:
    """Minimal genome schema for the baseline MA crossover strategy."""

    short_window: int
    long_window: int
    risk_fraction: float
    use_var_guard: bool = True
    use_drawdown_guard: bool = True

    def to_parameters(self) -> dict[str, float]:
        """Return parameters consumable by the trading strategy registry."""

        return {
            "short_window": float(self.short_window),
            "long_window": float(self.long_window),
            "risk_fraction": float(self.risk_fraction),
            "use_var_guard": 1.0 if self.use_var_guard else 0.0,
            "use_drawdown_guard": 1.0 if self.use_drawdown_guard else 0.0,
        }

    def to_decision_genome(self, identifier: str | None = None) -> DecisionGenome:
        """Convert the schema into the canonical :class:`DecisionGenome`."""

        genome_id = identifier or f"ma::{self.short_window}-{self.long_window}-{self.risk_fraction:.3f}"
        return new_genome(
            id=genome_id,
            parameters=self.to_parameters(),
            generation=0,
            species_type="ma_crossover",
        )

    @classmethod
    def from_decision_genome(cls, genome: DecisionGenome) -> "MovingAverageGenome":
        params = dict(getattr(genome, "parameters", {}) or {})
        return cls(
            short_window=int(params.get("short_window", 5)),
            long_window=int(params.get("long_window", 20)),
            risk_fraction=float(params.get("risk_fraction", 0.2)),
            use_var_guard=bool(params.get("use_var_guard", 1.0)),
            use_drawdown_guard=bool(params.get("use_drawdown_guard", 1.0)),
        )


@dataclass(slots=True)
class MovingAverageGenomeBounds:
    """Guardrails used during sampling, crossover, and mutation."""

    short_window: tuple[int, int] = (2, 30)
    long_window: tuple[int, int] = (20, 200)
    risk_fraction: tuple[float, float] = (0.05, 0.5)
    min_window_gap: int = 5

    def clamp(self, genome: MovingAverageGenome) -> MovingAverageGenome:
        short = int(np.clip(genome.short_window, *self.short_window))
        long = int(np.clip(genome.long_window, *self.long_window))
        if long - short < self.min_window_gap:
            long = min(self.long_window[1], max(long, short + self.min_window_gap))
            short = max(self.short_window[0], min(short, long - self.min_window_gap))
        risk = float(np.clip(genome.risk_fraction, *self.risk_fraction))
        return MovingAverageGenome(
            short_window=short,
            long_window=long,
            risk_fraction=risk,
            use_var_guard=genome.use_var_guard,
            use_drawdown_guard=genome.use_drawdown_guard,
        )

    def random(self, rng: random.Random) -> MovingAverageGenome:
        short = rng.randint(self.short_window[0], self.short_window[1])
        min_long = max(short + self.min_window_gap, self.long_window[0])
        long = rng.randint(min_long, self.long_window[1])
        risk = rng.uniform(*self.risk_fraction)
        return MovingAverageGenome(
            short_window=short,
            long_window=long,
            risk_fraction=risk,
            use_var_guard=rng.random() < 0.8,
            use_drawdown_guard=rng.random() < 0.8,
        )


@dataclass(slots=True)
class FitnessWeights:
    sharpe: float = 0.6
    sortino: float = 0.3
    max_drawdown: float = 0.4
    total_return: float = 0.2


@dataclass(slots=True)
class FitnessMetrics:
    fitness: float
    sharpe: float
    sortino: float
    max_drawdown: float
    total_return: float


@dataclass(slots=True)
class GARunConfig:
    population_size: int = 20
    generations: int = 15
    elite_count: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    seed: int | None = None
    bounds: MovingAverageGenomeBounds = field(default_factory=MovingAverageGenomeBounds)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    selection_mode: Literal["tournament", "mu_plus_lambda"] = "tournament"
    offspring_size: int | None = None

    def __post_init__(self) -> None:
        if self.population_size <= 1:
            raise ValueError("population_size must be greater than 1")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.elite_count < 1 or self.elite_count >= self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.selection_mode not in {"tournament", "mu_plus_lambda"}:
            raise ValueError("selection_mode must be 'tournament' or 'mu_plus_lambda'")
        if self.offspring_size is not None and self.offspring_size < 1:
            raise ValueError("offspring_size must be positive when provided")
        if self.selection_mode == "mu_plus_lambda" and self.population_size < 2:
            raise ValueError("mu+lambda mode requires population_size >= 2")


@dataclass(slots=True)
class GARunResult:
    best_genome: MovingAverageGenome
    best_metrics: FitnessMetrics
    population_history: list[tuple[MovingAverageGenome, FitnessMetrics]]

    def to_frame(self) -> pd.DataFrame:
        """Return a pandas DataFrame representing the best genome per generation."""

        rows = []
        for generation, (genome, metrics) in enumerate(self.population_history, start=1):
            rows.append(
                {
                    "generation": generation,
                    "short_window": genome.short_window,
                    "long_window": genome.long_window,
                    "risk_fraction": genome.risk_fraction,
                    "use_var_guard": genome.use_var_guard,
                    "use_drawdown_guard": genome.use_drawdown_guard,
                    "fitness": metrics.fitness,
                    "sharpe": metrics.sharpe,
                    "sortino": metrics.sortino,
                    "max_drawdown": metrics.max_drawdown,
                    "total_return": metrics.total_return,
                }
            )
        return pd.DataFrame(rows)

    def to_manifest(
        self,
        *,
        experiment: str,
        config: "GARunConfig",
        dataset_name: str,
        dataset_metadata: Mapping[str, Any] | None = None,
        code_version: str | None = None,
        notes: str | None = None,
        replay_command: str | None = None,
    ) -> dict[str, Any]:
        """Serialize the run into a reproducibility manifest."""

        leaderboard = [
            {
                "generation": generation,
                "genome": asdict(genome),
                "metrics": asdict(metrics),
            }
            for generation, (genome, metrics) in enumerate(
                self.population_history, start=1
            )
        ]

        manifest: dict[str, Any] = {
            "experiment": experiment,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "config": asdict(config),
            "dataset": {
                "name": dataset_name,
                "metadata": dict(dataset_metadata or {}),
            },
            "best_genome": asdict(self.best_genome),
            "best_metrics": asdict(self.best_metrics),
            "leaderboard": leaderboard,
            "generations": len(self.population_history),
        }
        if code_version:
            manifest["code_version"] = code_version
        if notes:
            manifest["notes"] = notes
        if replay_command:
            manifest["replay"] = {
                "command": replay_command,
                "seed": config.seed,
            }
        return manifest


def _compute_strategy_returns(prices: pd.Series, genome: MovingAverageGenome) -> pd.DataFrame:
    prices = prices.astype(float)
    df = pd.DataFrame({"price": prices})
    df["returns"] = df["price"].pct_change().fillna(0.0)
    df["short_ma"] = df["price"].rolling(genome.short_window).mean()
    df["long_ma"] = df["price"].rolling(genome.long_window).mean()
    df["signal"] = np.where(df["short_ma"] > df["long_ma"], 1.0, 0.0)
    df["position"] = df["signal"].shift(1).fillna(0.0) * genome.risk_fraction

    if genome.use_var_guard:
        df["var_guard"] = np.where(df["returns"].rolling(5).mean() < -0.015, 0.0, 1.0)
        df["position"] *= df["var_guard"]

    if genome.use_drawdown_guard:
        equity = (1 + df["position"] * df["returns"]).cumprod()
        drawdown = (equity.cummax() - equity) / equity.cummax().replace(0, np.nan)
        df["drawdown_guard"] = np.where(drawdown > 0.1, 0.0, 1.0)
        df["position"] *= df["drawdown_guard"]

    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity"] = (1.0 + df["strategy_returns"]).cumprod()
    return df


def evaluate_genome_fitness(
    prices: Sequence[float] | pd.Series,
    genome: MovingAverageGenome,
    *,
    weights: FitnessWeights | None = None,
) -> FitnessMetrics:
    """Evaluate a genome and return the weighted fitness metrics."""

    series = pd.Series(list(prices))
    data = _compute_strategy_returns(series, genome)
    returns = data["strategy_returns"].to_list()

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    if math.isinf(sortino):
        sortino = float(np.sign(sortino) * 10.0)
    if not math.isfinite(sortino):
        sortino = 0.0

    max_drawdown = calculate_max_drawdown(data["equity"])
    total_return = float(data["equity"].iat[-1] - 1.0)

    w = weights or FitnessWeights()
    fitness = (
        w.sharpe * sharpe
        + w.sortino * sortino
        + w.total_return * total_return
        - w.max_drawdown * max_drawdown
    )

    return FitnessMetrics(
        fitness=float(fitness),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_drawdown),
        total_return=float(total_return),
    )


def _crossover(
    parent_a: MovingAverageGenome,
    parent_b: MovingAverageGenome,
    rng: random.Random,
    bounds: MovingAverageGenomeBounds,
) -> MovingAverageGenome:
    short = parent_a.short_window if rng.random() < 0.5 else parent_b.short_window
    long = parent_a.long_window if rng.random() < 0.5 else parent_b.long_window
    risk = parent_a.risk_fraction if rng.random() < 0.5 else parent_b.risk_fraction
    use_var = parent_a.use_var_guard if rng.random() < 0.5 else parent_b.use_var_guard
    use_dd = parent_a.use_drawdown_guard if rng.random() < 0.5 else parent_b.use_drawdown_guard
    return bounds.clamp(
        MovingAverageGenome(
            short_window=short,
            long_window=long,
            risk_fraction=risk,
            use_var_guard=use_var,
            use_drawdown_guard=use_dd,
        )
    )


def _mutate(
    genome: MovingAverageGenome,
    rng: random.Random,
    bounds: MovingAverageGenomeBounds,
) -> MovingAverageGenome:
    delta_short = rng.randint(-3, 3)
    delta_long = rng.randint(-5, 5)
    delta_risk = rng.uniform(-0.05, 0.05)
    mutated = MovingAverageGenome(
        short_window=genome.short_window + delta_short,
        long_window=genome.long_window + delta_long,
        risk_fraction=genome.risk_fraction + delta_risk,
        use_var_guard=genome.use_var_guard if rng.random() > 0.1 else not genome.use_var_guard,
        use_drawdown_guard=genome.use_drawdown_guard if rng.random() > 0.1 else not genome.use_drawdown_guard,
    )
    return bounds.clamp(mutated)


def _breed_child(
    population: Sequence[tuple[MovingAverageGenome, FitnessMetrics]],
    rng: random.Random,
    bounds: MovingAverageGenomeBounds,
    *,
    crossover_rate: float,
    mutation_rate: float,
) -> MovingAverageGenome:
    parent_a = _select_parent(population, rng)
    parent_b = _select_parent(population, rng)
    child = parent_a
    if rng.random() < crossover_rate:
        child = _crossover(parent_a, parent_b, rng, bounds)
    if rng.random() < mutation_rate:
        child = _mutate(child, rng, bounds)
    return child


def _select_parent(
    population: Sequence[tuple[MovingAverageGenome, FitnessMetrics]],
    rng: random.Random,
) -> MovingAverageGenome:
    tournament = rng.sample(population, k=min(3, len(population)))
    best = max(tournament, key=lambda item: item[1].fitness)
    return best[0]


def run_ga_experiment(
    prices: Sequence[float] | pd.Series,
    config: GARunConfig | None = None,
) -> GARunResult:
    """Run an offline GA experiment on the provided price sequence."""

    cfg = config or GARunConfig()
    rng = random.Random(cfg.seed)
    bounds = cfg.bounds
    weights = cfg.weights

    population: List[MovingAverageGenome] = [bounds.random(rng) for _ in range(cfg.population_size)]

    evaluated: list[tuple[MovingAverageGenome, FitnessMetrics]] = []
    history: list[tuple[MovingAverageGenome, FitnessMetrics]] = []

    for _ in range(cfg.generations):
        evaluated = [
            (genome, evaluate_genome_fitness(prices, genome, weights=weights))
            for genome in population
        ]
        evaluated.sort(key=lambda item: item[1].fitness, reverse=True)
        if cfg.selection_mode == "mu_plus_lambda":
            parents_with_metrics = evaluated[: cfg.population_size]
            offspring_target = cfg.offspring_size or cfg.population_size
            offspring: list[MovingAverageGenome] = []
            while len(offspring) < offspring_target:
                offspring.append(
                    _breed_child(
                        parents_with_metrics,
                        rng,
                        bounds,
                        crossover_rate=cfg.crossover_rate,
                        mutation_rate=cfg.mutation_rate,
                    )
                )
            offspring_evaluated = [
                (child, evaluate_genome_fitness(prices, child, weights=weights))
                for child in offspring
            ]
            combined = parents_with_metrics + offspring_evaluated
            combined.sort(key=lambda item: item[1].fitness, reverse=True)
            history.append(combined[0])
            population = [genome for genome, _ in combined[: cfg.population_size]]
        else:
            history.append(evaluated[0])
            elites = [genome for genome, _ in evaluated[: cfg.elite_count]]
            new_population: list[MovingAverageGenome] = list(elites)
            while len(new_population) < cfg.population_size:
                new_population.append(
                    _breed_child(
                        evaluated,
                        rng,
                        bounds,
                        crossover_rate=cfg.crossover_rate,
                        mutation_rate=cfg.mutation_rate,
                    )
                )
            population = new_population

    best_genome, best_metrics = max(history, key=lambda item: item[1].fitness)
    return GARunResult(best_genome=best_genome, best_metrics=best_metrics, population_history=history)
