from __future__ import annotations

from src.evolution.catalogue_telemetry import build_catalogue_snapshot


def test_build_catalogue_snapshot_returns_none_without_catalogue() -> None:
    stats: dict[str, object] = {
        "seed_source": "factory",
        "population_size": 5,
        "generation": 0,
        "species_distribution": {"core": 5},
    }

    assert build_catalogue_snapshot(stats) is None


def test_build_catalogue_snapshot_normalises_payload() -> None:
    stats: dict[str, object] = {
        "seed_source": "catalogue",
        "population_size": 6,
        "generation": 3,
        "species_distribution": {"trend": "4", "mean_reversion": 2},
        "catalogue": {
            "name": "institutional_default",
            "version": "2025.09",
            "size": 5,
            "species": {"trend": 3, "carry": 2},
            "source_notes": ["derived", "roadmap"],
            "seeded_at": 1727000000.0,
            "entries": [
                {
                    "id": "catalogue/trend-alpha",
                    "name": "Trend Surfer Alpha",
                    "species": "trend_rider",
                    "generation": 4,
                    "tags": ["fx", "institutional"],
                    "performance_metrics": {"sharpe_ratio": 1.2, "cagr": 0.18},
                },
                {
                    "id": "catalogue/carry-beta",
                    "name": "Carry Flow Beta",
                    "species": "carry_arb",
                    "generation": 2,
                    "tags": ["fx"],
                    "performance_metrics": {"sharpe_ratio": 0.9},
                },
            ],
        },
    }

    snapshot = build_catalogue_snapshot(stats)
    assert snapshot is not None
    assert snapshot.catalogue_name == "institutional_default"
    assert snapshot.seed_source == "catalogue"
    assert snapshot.population_size == 6
    assert snapshot.catalogue_size == 5
    assert snapshot.catalogue_species == {"trend": 3, "carry": 2}
    assert snapshot.entries[0].identifier == "catalogue/trend-alpha"

    payload = snapshot.as_dict()
    assert payload["catalogue"]["entries"][0]["name"] == "Trend Surfer Alpha"
    assert payload["catalogue"]["entries"][0]["performance_metrics"]["sharpe_ratio"] == 1.2
