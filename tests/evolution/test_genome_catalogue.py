from src.genome.catalogue import load_default_catalogue


def test_default_catalogue_metadata_and_sampling():
    catalogue = load_default_catalogue()

    metadata = catalogue.metadata()
    assert metadata["name"] == "institutional_default"
    assert metadata["version"].startswith("2025")
    assert metadata["size"] >= 5
    assert "trend_rider" in metadata["species"]

    sample = catalogue.sample(4)
    assert len(sample) == 4
    ids = {genome.id for genome in sample}
    assert len(ids) == 4  # unique ids via sampling suffixes
    for genome in sample:
        assert isinstance(genome.parameters, dict)
        assert genome.parameters  # seeded parameters exist
        assert genome.species_type in metadata["species"]


def test_catalogue_describe_entries_includes_metrics():
    catalogue = load_default_catalogue()
    descriptions = catalogue.describe_entries()
    assert len(descriptions) == catalogue.metadata()["size"]
    for entry in descriptions:
        assert "id" in entry
        assert "species" in entry
        metrics = entry.get("performance_metrics", {})
        assert metrics
        assert "sharpe_ratio" in metrics
