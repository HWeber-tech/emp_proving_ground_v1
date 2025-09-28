#!/usr/bin/env python3
"""Generate Evolution Lab leaderboard artifacts aligned with the roadmap."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evolution.experiments.ma_crossover_ga import GARunConfig, run_ga_experiment
from src.evolution.experiments.reporting import render_leaderboard_markdown

DEFAULT_DATASET_NAME = "synthetic_trend_v1"
DEFAULT_LENGTH = 512
DEFAULT_POPULATION = 24
DEFAULT_GENERATIONS = 18
DEFAULT_ELITE = 4
DEFAULT_CROSSOVER = 0.7
DEFAULT_MUTATION = 0.25


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_output_dir() -> Path:
    return _repo_root() / "artifacts" / "evolution" / "ma_crossover"


def _resolve_git_revision(root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _generate_prices(length: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(loc=0.12, scale=0.45, size=length))
    prices = 100.0 + base
    return pd.Series(prices, name="price")


def _build_config(args: argparse.Namespace) -> GARunConfig:
    return GARunConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )


def _build_replay_command(args: argparse.Namespace) -> str:
    parts = ["python", "scripts/generate_evolution_lab.py"]
    if args.seed is not None:
        parts.extend(["--seed", str(args.seed)])
    if args.length != DEFAULT_LENGTH:
        parts.extend(["--length", str(args.length)])
    if args.population_size != DEFAULT_POPULATION:
        parts.extend(["--population-size", str(args.population_size)])
    if args.generations != DEFAULT_GENERATIONS:
        parts.extend(["--generations", str(args.generations)])
    if args.elite_count != DEFAULT_ELITE:
        parts.extend(["--elite-count", str(args.elite_count)])
    if abs(args.crossover_rate - DEFAULT_CROSSOVER) > 1e-9:
        parts.extend(["--crossover-rate", str(args.crossover_rate)])
    if abs(args.mutation_rate - DEFAULT_MUTATION) > 1e-9:
        parts.extend(["--mutation-rate", str(args.mutation_rate)])
    return " ".join(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=_repo_root() / "docs" / "research" / "evolution_lab.md",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE)
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER)
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    return parser.parse_args()


def _manifest_notes() -> str:
    return (
        "Synthetic trending price series approximating Tier-0 bootstrap conditions. "
        "Use for regression testing of GA improvements."
    )


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = _generate_prices(args.length, args.seed)
    config = _build_config(args)
    result = run_ga_experiment(prices, config)

    manifest = result.to_manifest(
        experiment="ma_crossover_ga",
        config=config,
        dataset_name=args.dataset_name,
        dataset_metadata={
            "length": len(prices),
            "seed": args.seed,
            "mean": float(prices.mean()),
            "std": float(prices.std(ddof=0)),
        },
        code_version=_resolve_git_revision(repo_root),
        notes=_manifest_notes(),
        replay_command=_build_replay_command(args),
    )

    history_frame = result.to_frame()

    manifest_path = output_dir / "manifest.json"
    prices_path = output_dir / "dataset.csv"
    leaderboard_path = output_dir / "generation_history.csv"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    prices.to_csv(prices_path, index=False)
    history_frame.to_csv(leaderboard_path, index=False)

    doc_path = args.docs_path
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard_md = render_leaderboard_markdown([manifest])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    doc_content = (
        "# Evolution Lab Leaderboard\n\n"
        f"_Auto-generated on {timestamp} using `scripts/generate_evolution_lab.py`._\n\n"
        "## Current Experiments\n\n"
        f"{leaderboard_md}\n"
        "## Reproducibility Artifacts\n\n"
        f"- Manifest: `{manifest_path.relative_to(repo_root)}`\n"
        f"- Dataset: `{prices_path.relative_to(repo_root)}`\n"
        f"- Generation history: `{leaderboard_path.relative_to(repo_root)}`\n\n"
        "## Follow-on Backlog\n\n"
        "- [ ] Introduce speciation and diversity preservation experiments.\n"
        "- [ ] Evaluate Pareto-front selection for multi-objective fitness.\n"
        "- [ ] Swap synthetic datasets with live market snapshots for benchmarking.\n"
        "- [ ] Automate nightly leaderboard refresh with CI artifact publishing.\n"
        "- [ ] Integrate promoted genomes into the strategy registry feature flags.\n"
    )

    doc_path.write_text(doc_content)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Updated leaderboard at {doc_path}")


if __name__ == "__main__":
    main()
