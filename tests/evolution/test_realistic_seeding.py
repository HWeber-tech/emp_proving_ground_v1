from __future__ import annotations

import random

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.seeding import RealisticGenomeSeeder


def test_realistic_genome_seeder_cycles_species():
    rng = random.Random(7)
    seeder = RealisticGenomeSeeder(rng=rng)

    seeds = [seeder.sample() for _ in range(len(seeder.templates) * 2)]
    species = [seed.species for seed in seeds]

    # Ensure we cycle through the catalogue templates before repeating
    assert species[0] == species[len(seeder.templates)]
    assert len(set(species)) >= 3


def test_evolution_engine_default_population_has_lineage_metadata():
    engine = EvolutionEngine(EvolutionConfig(population_size=6, elite_count=2))
    engine._rng.seed(11)

    population = engine.ensure_population()

    assert len(population) == 6
    species = {getattr(genome, "species_type", None) for genome in population}
    assert len(species) >= 3

    for genome in population:
        parent_ids = getattr(genome, "parent_ids", [])
        assert isinstance(parent_ids, list)
        assert parent_ids  # catalogue-inspired seeds include provenance

        mutation_history = getattr(genome, "mutation_history", [])
        assert isinstance(mutation_history, list)
        assert mutation_history

        metrics = getattr(genome, "performance_metrics", {})
        assert isinstance(metrics, dict)
        assert metrics

        metadata = getattr(genome, "metadata", {})
        assert isinstance(metadata, dict)
        assert metadata.get("seed_name")
        assert metadata.get("seed_species")


def test_evolution_engine_catalogue_population_reports_seed_metadata():
    engine = EvolutionEngine(
        EvolutionConfig(population_size=5, elite_count=2, use_catalogue=True)
    )
    engine._rng.seed(23)

    population = engine.ensure_population()
    assert len(population) == 5

    stats = engine.get_population_statistics()
    assert stats["seed_source"] == "catalogue"

    seed_metadata = stats.get("seed_metadata")
    assert isinstance(seed_metadata, dict)
    assert seed_metadata.get("seed_names")
    assert seed_metadata.get("seed_tags")
    assert seed_metadata.get("seed_species")

    templates = seed_metadata.get("seed_templates")
    assert isinstance(templates, list)
    assert templates
    top_template = templates[0]
    assert top_template.get("name")
    assert 0.0 < float(top_template.get("share", 0.0)) <= 1.0
