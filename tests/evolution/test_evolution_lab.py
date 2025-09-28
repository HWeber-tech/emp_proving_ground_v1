import pandas as pd

from src.evolution.experiments.evolution_lab import (
    GARunConfig,
    render_evolution_lab_markdown,
    run_ma_crossover_lab,
)
from src.governance.strategy_registry import StrategyRegistry


def _trend_series(length: int = 96, seed: int = 99) -> pd.Series:
    rng = pd.Series(pd.Index(range(length)))
    noise = pd.Series([((i * 0.05) + (seed % 7) * 0.01) for i in range(length)])
    return 100 + rng * 0.2 + noise


def test_run_ma_crossover_lab_produces_manifest_and_leaderboard() -> None:
    series = _trend_series(128)
    config = GARunConfig(population_size=10, generations=4, elite_count=2, seed=42)

    report = run_ma_crossover_lab(
        series,
        experiment_name="unit-test",
        dataset_name="synthetic",
        config=config,
    )

    assert report.manifest.experiment_name == "unit-test"
    assert report.manifest.sample_size == 128
    assert len(report.leaderboard) == config.generations

    markdown = render_evolution_lab_markdown(report)
    assert "# Evolution Lab Leaderboard" in markdown
    assert "Generation" in markdown


def test_run_ma_crossover_lab_registers_when_enabled(tmp_path) -> None:
    series = _trend_series(64)
    config = GARunConfig(population_size=8, generations=3, elite_count=2, seed=7)
    registry_path = tmp_path / "registry.db"

    report = run_ma_crossover_lab(
        series,
        experiment_name="register-test",
        dataset_name="synthetic",
        config=config,
        register_champion=True,
        registry_db_path=registry_path,
    )

    assert report.registered_champion is True
    assert report.registry_path == registry_path

    registry = StrategyRegistry(str(registry_path))
    try:
        cursor = registry.conn.cursor()
        cursor.execute("SELECT fitness_score, genome_id FROM strategies LIMIT 1")
        row = cursor.fetchone()
        assert row is not None
        assert row["fitness_score"] is not None
        assert isinstance(row["genome_id"], str) and row["genome_id"]
    finally:
        registry.close()
