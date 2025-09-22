import pytest

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine


@pytest.fixture(autouse=True)
def _clear_flag_env(monkeypatch):
    monkeypatch.delenv("EVOLUTION_USE_CATALOGUE", raising=False)


def test_evolution_engine_uses_catalogue_when_enabled():
    engine = EvolutionEngine(EvolutionConfig(population_size=6, elite_count=2, use_catalogue=True))

    population = engine.ensure_population()
    assert len(population) == 6
    species = {getattr(genome, "species_type", None) for genome in population}
    assert species  # species seeded from catalogue

    stats = engine.get_population_statistics()
    assert stats["seed_source"] == "catalogue"
    assert "catalogue" in stats
    assert stats["catalogue"]["name"] == "institutional_default"
    assert stats["catalogue"]["seeded_at"] is not None


def test_evolution_engine_obeys_env_feature_flag(monkeypatch):
    monkeypatch.setenv("EVOLUTION_USE_CATALOGUE", "true")
    engine = EvolutionEngine(EvolutionConfig(population_size=4, elite_count=1))

    population = engine.ensure_population()
    assert len(population) == 4

    stats = engine.get_population_statistics()
    assert stats["seed_source"] == "catalogue"
    assert stats["catalogue"]["version"].startswith("2025")
    assert stats["catalogue"]["seeded_at"] is not None
