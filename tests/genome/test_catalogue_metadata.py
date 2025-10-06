from __future__ import annotations

from src.genome.catalogue import load_default_catalogue


def test_catalogue_sample_embeds_seed_metadata() -> None:
    catalogue = load_default_catalogue()
    genomes = catalogue.sample(len(catalogue.entries), shuffle=False)

    for entry, genome in zip(catalogue.entries, genomes, strict=True):
        metadata = getattr(genome, "metadata", {}) or {}

        assert metadata.get("seed_name") == entry.name
        assert metadata.get("seed_species") == entry.species
        assert metadata.get("seed_catalogue_id") == entry.identifier
        assert metadata.get("seed_catalogue_generation") == entry.generation
        assert metadata.get("seed_tags") == list(entry.tags)
        assert metadata.get("seed_parent_ids") == list(entry.parent_ids)
        assert metadata.get("seed_mutation_history") == list(entry.mutation_history)

