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
