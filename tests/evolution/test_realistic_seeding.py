from __future__ import annotations

import random

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.seeding import RealisticGenomeSeeder, load_experiment_seed_templates
from src.genome.catalogue import load_default_catalogue


def test_realistic_genome_seeder_cycles_species():
    rng = random.Random(7)
    seeder = RealisticGenomeSeeder(rng=rng)

    seeds = [seeder.sample() for _ in range(len(seeder.templates) * 2)]
    species = [seed.species for seed in seeds]

    # Ensure we cycle through the catalogue templates before repeating
    assert species[0] == species[len(seeder.templates)]
    assert len(set(species)) >= 3

    assert all(seed.catalogue_entry_id for seed in seeds)


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
        assert metadata.get("seed_catalogue_id")
        assert metadata.get("seed_parent_ids")
        assert metadata.get("seed_mutation_history")
        assert metadata.get("seed_performance_metrics")

    stats = engine.get_population_statistics()
    seed_metadata = stats.get("seed_metadata") if isinstance(stats, dict) else None
    assert seed_metadata is not None
    assert seed_metadata.get("seed_catalogue_ids")
    assert seed_metadata.get("seed_parent_ids")
    assert seed_metadata.get("seed_mutations")


def test_experiment_templates_loaded_from_artifacts():
    templates = load_experiment_seed_templates()
    assert templates, "expected experiment templates from artifacts"

    artifact_templates = [
        template
        for template in templates
        if template.catalogue_entry_id and template.catalogue_entry_id.startswith("artifact/")
    ]
    assert artifact_templates, "expected at least one artifact-derived template"

    seeder = RealisticGenomeSeeder()
    catalogue_ids = {
        template.catalogue_entry_id for template in seeder.templates if template.catalogue_entry_id
    }
    assert any(identifier.startswith("artifact/") for identifier in catalogue_ids)

    sample_seed = artifact_templates[0].spawn(random.Random(3))
    assert sample_seed.catalogue_entry_id and sample_seed.catalogue_entry_id.startswith("artifact/")
    assert sample_seed.parameters
    assert any(tag.startswith("experiment:") for tag in sample_seed.tags)


def test_catalogue_entries_emit_seed_metadata():
    catalogue = load_default_catalogue()
    assert catalogue.entries, "expected catalogue entries for metadata coverage"

    genome = catalogue.entries[0].instantiate()

    metadata = getattr(genome, "metadata", {})
    assert metadata
    assert metadata["seed_source"] == "catalogue"
    assert metadata["seed_name"] == catalogue.entries[0].name
    assert metadata["seed_catalogue_id"] == catalogue.entries[0].identifier
    assert metadata["seed_species"] == catalogue.entries[0].species
    assert metadata.get("seed_tags")
    assert metadata.get("seed_parent_ids")
