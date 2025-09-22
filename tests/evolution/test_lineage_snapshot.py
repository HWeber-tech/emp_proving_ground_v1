from __future__ import annotations

from src.core.evolution.engine import EvolutionSummary
from src.orchestration.evolution_cycle import ChampionRecord, FitnessReport
from src.evolution.lineage_telemetry import EvolutionLineageSnapshot, build_lineage_snapshot


def test_build_lineage_snapshot_fuses_stats_and_champion_metadata():
    report = FitnessReport(
        fitness_score=1.23,
        max_drawdown=0.05,
        sharpe_ratio=1.1,
        total_return=0.42,
        volatility=0.2,
        metadata={"desk": "trend", "evaluated": True},
    )
    champion = ChampionRecord(
        genome_id="core-evo-0001",
        fitness=1.23,
        report=report,
        registered=True,
        parent_ids=("core-evo-0000",),
        species="trend_following",
        mutation_history=("g1:mutation:drift", "g1:mutation:risk"),
    )
    stats = {
        "generation": 5,
        "species_distribution": {"trend_following": 3, "carry": 2},
        "seed_source": "catalogue",
        "catalogue": {"name": "institutional", "seeded_at": 1234.0},
    }
    summary = EvolutionSummary(
        generation=6,
        population_size=5,
        best_fitness=1.5,
        average_fitness=1.1,
        elite_count=2,
        timestamp=111.0,
    )

    snapshot = build_lineage_snapshot(stats, champion, summary=summary)

    assert isinstance(snapshot, EvolutionLineageSnapshot)
    assert snapshot.generation == 6
    assert snapshot.champion_id == "core-evo-0001"
    assert snapshot.parent_ids == ("core-evo-0000",)
    assert snapshot.species == "trend_following"
    assert snapshot.seed_source == "catalogue"
    assert snapshot.species_distribution["trend_following"] == 3
    assert snapshot.catalogue["name"] == "institutional"

    payload = snapshot.as_dict(max_parents=1, max_mutations=1)
    assert payload["champion"]["parent_ids"] == ["core-evo-0000"]
    assert payload["champion"]["mutation_history"] == ["g1:mutation:drift"]
    assert payload["champion"]["metadata"]["desk"] == "trend"
    assert payload["population"]["seed_source"] == "catalogue"

    fingerprint = snapshot.fingerprint()
    assert fingerprint[0] == 6
    assert snapshot.to_markdown().startswith("### Evolution lineage")


def test_build_lineage_snapshot_returns_none_without_champion():
    stats = {"generation": 2, "seed_source": "factory"}

    assert build_lineage_snapshot(stats, None) is None
