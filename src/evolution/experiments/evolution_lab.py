"""Evolution Lab orchestration helpers.

This module wraps the moving-average GA experiment with additional
book-keeping so the results can be published into the "Evolution Lab"
backlog described in the high-impact roadmap.

Responsibilities
----------------
* Execute deterministic GA experiments with manifest metadata (dataset
  fingerprint, configuration, seeds).
* Optionally register the best genome in the persistent strategy
  registry behind a feature flag so supervisors can promote the
  champion into paper trading.
* Provide helpers to render Markdown leaderboards that land under
  ``docs/research/evolution_lab.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evolution.experiments.ma_crossover_ga import (
    GARunConfig,
    GARunResult,
    MovingAverageGenome,
    run_ga_experiment,
)
from src.governance.strategy_registry import StrategyRegistry

__all__ = [
    "EvolutionLabManifest",
    "EvolutionLabLeaderboardEntry",
    "EvolutionLabReport",
    "run_ma_crossover_lab",
    "render_evolution_lab_markdown",
]


@dataclass(slots=True)
class EvolutionLabManifest:
    """Metadata describing a deterministic Evolution Lab run."""

    experiment_name: str
    dataset_name: str
    dataset_hash: str
    sample_size: int
    seed: int | None
    generated_at: str
    config: dict[str, float]
    notes: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        """Return a stable JSON representation for persistence/tests."""

        payload = {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "generated_at": self.generated_at,
            "config": self.config,
            "notes": self.notes,
        }
        return json.dumps(payload, sort_keys=True)


@dataclass(slots=True)
class EvolutionLabLeaderboardEntry:
    """Top genome snapshot captured after each generation."""

    generation: int
    genome: MovingAverageGenome
    fitness: float
    sharpe: float
    sortino: float
    max_drawdown: float
    total_return: float

    def as_row(self) -> list[str]:
        """Return a Markdown-ready row."""

        return [
            str(self.generation),
            str(self.genome.short_window),
            str(self.genome.long_window),
            f"{self.genome.risk_fraction:.3f}",
            "Yes" if self.genome.use_var_guard else "No",
            "Yes" if self.genome.use_drawdown_guard else "No",
            f"{self.fitness:.4f}",
            f"{self.sharpe:.3f}",
            f"{self.sortino:.3f}",
            f"{self.max_drawdown:.3f}",
            f"{self.total_return:.3f}",
        ]


@dataclass(slots=True)
class EvolutionLabReport:
    """Full report returned by :func:`run_ma_crossover_lab`."""

    manifest: EvolutionLabManifest
    leaderboard: list[EvolutionLabLeaderboardEntry]
    ga_result: GARunResult
    registered_champion: bool
    registry_path: Path | None


def _fingerprint_prices(series: pd.Series) -> str:
    cleaned = series.fillna(0.0).astype("float64")
    digest = hashlib.sha256(cleaned.to_numpy().tobytes()).hexdigest()
    return digest


def _normalise_series(prices: Sequence[float] | pd.Series) -> pd.Series:
    if isinstance(prices, pd.Series):
        series = prices.copy()
    else:
        series = pd.Series(list(prices))
    if series.empty:
        raise ValueError("prices must contain at least one element")
    series = series.astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("prices must contain at least one finite element")
    return series


def _should_register(value: bool | None) -> bool:
    if value is not None:
        return value
    raw = os.environ.get("EVOLUTION_LAB_REGISTER_CHAMPION")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _serialise_config(config: GARunConfig) -> dict[str, float]:
    # ``asdict`` flattens nested dataclasses; we only need primitive config
    payload = asdict(config)
    payload.pop("bounds", None)
    payload.pop("weights", None)
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in payload.items()}


def _build_leaderboard(result: GARunResult) -> list[EvolutionLabLeaderboardEntry]:
    entries: list[EvolutionLabLeaderboardEntry] = []
    for generation, (genome, metrics) in enumerate(result.population_history, start=1):
        entries.append(
            EvolutionLabLeaderboardEntry(
                generation=generation,
                genome=genome,
                fitness=metrics.fitness,
                sharpe=metrics.sharpe,
                sortino=metrics.sortino,
                max_drawdown=metrics.max_drawdown,
                total_return=metrics.total_return,
            )
        )
    return entries


def run_ma_crossover_lab(
    prices: Sequence[float] | pd.Series,
    *,
    experiment_name: str,
    dataset_name: str,
    config: GARunConfig | None = None,
    register_champion: bool | None = None,
    registry_db_path: str | Path | None = None,
    notes: Mapping[str, object] | None = None,
) -> EvolutionLabReport:
    """Execute the GA experiment with Evolution Lab bookkeeping."""

    series = _normalise_series(prices)
    cfg = config or GARunConfig()
    result = run_ga_experiment(series, cfg)

    manifest = EvolutionLabManifest(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        dataset_hash=_fingerprint_prices(series),
        sample_size=int(series.shape[0]),
        seed=cfg.seed,
        generated_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        config=_serialise_config(cfg),
        notes=dict(notes or {}),
    )

    leaderboard = _build_leaderboard(result)

    should_register = _should_register(register_champion)
    registry_path: Path | None = None
    registered = False
    if should_register:
        registry_path = Path(registry_db_path) if registry_db_path else Path("governance.db")
        registry = StrategyRegistry(str(registry_path))
        try:
            champion = result.best_genome
            metrics = result.best_metrics
            decision_genome = champion.to_decision_genome(
                identifier=f"evolution-lab::{manifest.dataset_hash[:8]}::{manifest.generated_at}"
            )
            fitness_report = {
                "fitness_score": metrics.fitness,
                "sharpe_ratio": metrics.sharpe,
                "sortino_ratio": metrics.sortino,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "metadata": {
                    "evolution_lab": {
                        "manifest": json.loads(manifest.to_json()),
                    }
                },
            }
            provenance = {
                "seed_source": "evolution_lab",
                "experiment": {
                    "name": experiment_name,
                    "dataset": dataset_name,
                    "generated_at": manifest.generated_at,
                },
            }
            registered = registry.register_champion(
                decision_genome,
                fitness_report,
                provenance=provenance,
            )
        finally:
            registry.close()

    return EvolutionLabReport(
        manifest=manifest,
        leaderboard=leaderboard,
        ga_result=result,
        registered_champion=registered,
        registry_path=registry_path,
    )


def render_evolution_lab_markdown(report: EvolutionLabReport) -> str:
    """Render a Markdown document summarising the experiment."""

    manifest = report.manifest
    header = [
        "# Evolution Lab Leaderboard",
        "",
        f"*Last generated:* {manifest.generated_at}",
        "",
        "## Experiment Manifest",
        "",
        f"- **Experiment:** `{manifest.experiment_name}`",
        f"- **Dataset:** `{manifest.dataset_name}`",
        f"- **Dataset hash:** `{manifest.dataset_hash}`",
        f"- **Sample size:** {manifest.sample_size}",
        f"- **Seed:** {manifest.seed if manifest.seed is not None else 'null'}",
        f"- **Config:** `{json.dumps(manifest.config, sort_keys=True)}`",
    ]
    if manifest.notes:
        header.append("- **Notes:**")
        for key, value in manifest.notes.items():
            header.append(f"  - {key}: {value}")

    header.extend(["", "## Generation Leaderboard", ""])

    columns = [
        "Generation",
        "Short",
        "Long",
        "Risk Fraction",
        "VaR Guard",
        "Drawdown Guard",
        "Fitness",
        "Sharpe",
        "Sortino",
        "Max Drawdown",
        "Total Return",
    ]

    table_lines = ["| " + " | ".join(columns) + " |"]
    table_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for entry in report.leaderboard:
        table_lines.append("| " + " | ".join(entry.as_row()) + " |")

    footer = [""]
    if report.registered_champion:
        footer.append(
            "Champion genome registered in strategy registry"
            f" at `{report.registry_path}`."
        )

    return "\n".join(header + table_lines + footer) + "\n"
