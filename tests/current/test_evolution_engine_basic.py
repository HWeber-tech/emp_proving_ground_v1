import pytest

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine, EvolutionSummary


def test_evolution_engine_emits_summary_and_tracks_generation():
    engine = EvolutionEngine(EvolutionConfig(population_size=6, elite_count=2))

    first_summary = engine.evolve()
    assert isinstance(first_summary, EvolutionSummary)
    assert first_summary.population_size == 6
    assert first_summary.elite_count >= 0

    second_summary = engine.evolve()
    assert second_summary.generation == first_summary.generation + 1
    assert second_summary.population_size == 6


def test_mu_plus_lambda_mode_expands_population_and_preserves_survivors():
    config = EvolutionConfig(
        population_size=4,
        elite_count=2,
        selection_mode="mu_plus_lambda",
        offspring_count=3,
    )
    engine = EvolutionEngine(config)
    engine._rng.seed(21)  # type: ignore[attr-defined]

    initial_population = engine.ensure_population()
    for idx, genome in enumerate(initial_population, start=1):
        setattr(genome, "fitness", float(idx))

    summary = engine.evolve()
    assert summary.population_size == 7

    population_after_first = engine.get_population()
    assert len(population_after_first) == 7

    original_ids = {str(getattr(genome, "id", "")) for genome in initial_population}
    assert original_ids.issubset({str(getattr(genome, "id", "")) for genome in population_after_first})

    for genome in population_after_first:
        if str(getattr(genome, "id", "")) in original_ids:
            setattr(genome, "fitness", 100.0)
        else:
            setattr(genome, "fitness", 0.1)

    next_summary = engine.evolve()
    assert next_summary.population_size == 7

    population_after_second = engine.get_population()
    survivor_ids = {str(getattr(genome, "id", "")) for genome in population_after_second}
    assert original_ids.issubset(survivor_ids)


def test_evolution_config_rejects_invalid_offspring_count():
    with pytest.raises(ValueError):
        EvolutionConfig(offspring_count=-1)
