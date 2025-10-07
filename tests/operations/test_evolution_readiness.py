from __future__ import annotations

from datetime import datetime, timezone

from src.evolution.lineage_telemetry import EvolutionLineageSnapshot
from src.operations.evolution_readiness import (
    EvolutionReadinessStatus,
    evaluate_evolution_readiness,
)


def _build_lineage(*, registered: bool) -> EvolutionLineageSnapshot:
    return EvolutionLineageSnapshot(
        generation=3,
        champion_id="core-evo-00042",
        fitness=1.23,
        registered=registered,
        species="trend_rider",
        parent_ids=("seed-001", "seed-002"),
        mutation_history=("seed", "mutation-a"),
        seed_source="realistic_sampler",
        species_distribution={"trend_rider": 4, "mean_reversion": 2},
        evaluation_metadata={"fitness_score": 1.23},
        catalogue={"name": "institutional", "version": "2024.1"},
        summary={"generation": 3, "best_fitness": 1.23},
        seed_metadata={
            "seed_names": {"Trend Surfer Alpha": 4},
            "seed_templates": [
                {"name": "Trend Surfer Alpha", "count": 4, "share": 0.5},
                {"name": "Sigma Reversion", "count": 2, "share": 0.25},
            ],
            "seed_parent_ids": {"desk-trend-2019": 4},
            "seed_mutations": {"g0:seed:trend-alpha": 4},
        },
    )


def _seed_stats() -> dict[str, object]:
    return {
        "seed_source": "realistic_sampler",
        "seed_metadata": {
            "seed_names": {"Trend Surfer Alpha": 4, "Sigma Reversion": 2},
            "seed_templates": [
                {"name": "Trend Surfer Alpha", "count": 4, "share": 0.5},
                {"name": "Sigma Reversion", "count": 2, "share": 0.25},
            ],
            "seed_parent_ids": {"desk-trend-2019": 4, "desk-meanrev-2021": 2},
            "seed_mutations": {
                "g0:seed:trend-alpha": 4,
                "g0:seed:mean-reversion": 2,
            },
        },
    }


def test_readiness_review_when_flag_disabled() -> None:
    stats = _seed_stats()
    lineage = _build_lineage(registered=False)

    snapshot = evaluate_evolution_readiness(
        adaptive_runs_enabled=False,
        population_stats=stats,
        lineage_snapshot=lineage,
        now=datetime(2024, 3, 11, tzinfo=timezone.utc),
    )

    assert snapshot.status is EvolutionReadinessStatus.review
    assert snapshot.adaptive_runs_enabled is False
    assert set(snapshot.seed_templates) >= {"Trend Surfer Alpha", "Sigma Reversion"}
    assert "Adaptive runs disabled" in " ".join(snapshot.issues)
    assert snapshot.metadata["seed_metadata"]["seed_names"]["Trend Surfer Alpha"] == 4
    assert snapshot.champion_registered is False


def test_readiness_ready_when_flag_enabled_and_registered() -> None:
    stats = _seed_stats()
    lineage = _build_lineage(registered=True)

    snapshot = evaluate_evolution_readiness(
        adaptive_runs_enabled=True,
        population_stats=stats,
        lineage_snapshot=lineage,
        now=datetime(2024, 3, 11, tzinfo=timezone.utc),
    )

    assert snapshot.status is EvolutionReadinessStatus.ready
    assert snapshot.adaptive_runs_enabled is True
    assert snapshot.champion_id == "core-evo-00042"
    assert snapshot.champion_registered is True
    assert snapshot.lineage_generation == 3


def test_readiness_blocked_without_seed_metadata() -> None:
    stats = {"seed_source": None}
    lineage = _build_lineage(registered=False)

    snapshot = evaluate_evolution_readiness(
        adaptive_runs_enabled=False,
        population_stats=stats,
        lineage_snapshot=lineage,
        now=datetime(2024, 3, 11, tzinfo=timezone.utc),
    )

    assert snapshot.status is EvolutionReadinessStatus.blocked
    assert any("Seed source" in issue for issue in snapshot.issues)


def test_readiness_markdown_includes_key_details() -> None:
    stats = _seed_stats()
    lineage = _build_lineage(registered=True)

    snapshot = evaluate_evolution_readiness(
        adaptive_runs_enabled=True,
        population_stats=stats,
        lineage_snapshot=lineage,
        now=datetime(2024, 3, 11, tzinfo=timezone.utc),
    )

    markdown = snapshot.to_markdown()
    assert "Evolution readiness" in markdown
    assert "core-evo-00042" in markdown
    assert "Seed templates" in markdown
    payload = snapshot.as_dict()
    assert payload["status"] == "ready"
    assert payload["adaptive_runs_enabled"] is True
