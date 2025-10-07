import math
from typing import cast

import pytest

from src.core.population_manager import PopulationManager
from src.genome.models.genome import DecisionGenome


def _legacy_factory_closure():
    counter = {"i": 0}

    def factory():
        i = counter["i"]
        counter["i"] += 1
        # Legacy dict/object variant (dict-like)
        return {
            "id": f"L{i}",
            "parameters": {
                "risk_tolerance": 0.5,
                "position_size_factor": 0.05,
                "stop_loss_factor": 0.01,
                "take_profit_factor": 0.02,
                "trend_sensitivity": 0.7,
                "momentum_window": 10,  # int-like (will coerce to float)
            },
            "generation": -1,
            "species_type": "trading_strategy",
        }

    return factory


def test_create_and_store_genome():
    pm = PopulationManager(population_size=2, cache_ttl=1)
    factory = _legacy_factory_closure()
    pm.initialize_population(factory)

    pop = pm.get_population()
    assert len(pop) == 2
    # Should be canonical instances internally
    g0 = pm.get_genome_by_id("L0")
    g1 = pm.get_genome_by_id("L1")
    assert isinstance(g0, DecisionGenome)
    assert isinstance(g1, DecisionGenome)
    assert g0.parameters["momentum_window"] == pytest.approx(10.0)


def test_update_fitness_and_generation():
    pm = PopulationManager(population_size=5, cache_ttl=1)
    pm.initialize_population(_legacy_factory_closure())

    # Evolve with simple metrics
    metrics = {
        "total_return": 0.2,
        "max_drawdown": 0.1,
        "win_rate": 0.6,
        "sharpe_ratio": 1.2,
    }
    pm.evolve_population(market_data={}, performance_metrics=metrics)

    assert pm.generation == 1
    # All genomes should have non-negative fitness
    for g in pm.get_population():
        assert (g.fitness or 0.0) >= 0.0
        assert isinstance(g.fitness, float) or g.fitness is None  # canonical allows None pre-eval


def test_mutation_flow_integrates():
    pm = PopulationManager(population_size=1, cache_ttl=1)
    pm.initialize_population(_legacy_factory_closure())

    pm.generation = 2  # ensure generation tag in mutations

    original = cast(DecisionGenome, pm.get_population()[0])
    mutated = pm._mutate(original, mutation_rate=1.0)  # force mutations

    # Check canonical mutation tags
    assert any(tag == "g2:mutation:gaussian" for tag in mutated.mutation_history)
    assert any(tag.startswith("g2:") and "->" in tag for tag in mutated.mutation_history)

    # Parameters likely changed due to mutation_rate=1.0
    changed = False
    for k, v in original.parameters.items():
        if not math.isclose(float(v), float(mutated.parameters.get(k, float(v)))):
            changed = True
            break
    assert changed


def test_generate_initial_population_uses_realistic_seeds():
    pm = PopulationManager(population_size=3, cache_ttl=1)

    pm.population.clear()
    pm._catalogue_flag = False  # type: ignore[attr-defined]
    pm._catalogue = None  # type: ignore[attr-defined]

    pm._generate_initial_population()

    assert len(pm.population) == 3
    stats = pm.get_population_statistics()
    assert stats["seed_source"] == "realistic_sampler"
    seed_metadata = stats.get("seed_metadata")
    assert seed_metadata is not None
    assert seed_metadata.get("seed_names")
    for genome in pm.population:
        metadata = getattr(genome, "metadata", {}) or {}
        assert metadata.get("seed_name")
        assert metadata.get("seed_species")


def test_generate_initial_population_factory_fallback(monkeypatch):
    pm = PopulationManager(population_size=3, cache_ttl=1)

    pm.population.clear()
    pm._catalogue_flag = False  # type: ignore[attr-defined]
    pm._catalogue = None  # type: ignore[attr-defined]
    pm._seed_sampler = None  # type: ignore[attr-defined]

    monkeypatch.setattr(pm, "_seed_with_sampler", lambda provider: ([], None))

    pm._generate_initial_population()

    assert len(pm.population) == 3
    stats = pm.get_population_statistics()
    assert stats["seed_source"] == "factory"
    assert stats["population_size"] == 3


def test_catalogue_initialization_records_seed_metadata():
    pm = PopulationManager(population_size=3, cache_ttl=1, use_catalogue=True)
    pm.initialize_population(_legacy_factory_closure())

    stats = pm.get_population_statistics()
    assert stats["seed_source"] == "catalogue"

    seed_metadata = stats.get("seed_metadata")
    assert seed_metadata is not None
    assert seed_metadata["total_seeded"] == 3
    assert seed_metadata["seed_names"]
    assert seed_metadata["seed_catalogue_ids"]
