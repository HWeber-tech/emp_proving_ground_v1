import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.governance.strategy_registry import StrategyRegistry


def _build_provenance() -> dict[str, object]:
    return {
        "seed_source": "catalogue",
        "generation": 4,
        "population_size": 6,
        "catalogue": {
            "name": "institutional_default",
            "version": "2025.09",
            "size": 5,
            "species": {"trend": 3, "carry": 2},
            "seeded_at": 1_726_000_000.0,
            "source_notes": ["calibrated desk catalogue"],
        },
        "entry": {
            "id": "catalogue/trend-alpha",
            "name": "Trend Surfer Alpha",
            "species": "trend_rider",
            "generation": 4,
            "tags": ["fx", "swing"],
            "performance_metrics": {"sharpe_ratio": 1.3},
        },
    }


@pytest.mark.parametrize("status", ["evolved", "approved"])
def test_strategy_registry_records_catalogue_provenance(tmp_path, status: str) -> None:
    registry = StrategyRegistry(db_path=str(tmp_path / "registry.db"))
    genome = SimpleNamespace(
        id="catalogue/trend-alpha::0",
        decision_tree={"nodes": 12},
        name="trend-alpha",
        generation=4,
    )
    fitness_report = {
        "fitness_score": 1.42,
        "max_drawdown": -0.11,
        "sharpe_ratio": 1.2,
        "total_return": 0.22,
        "volatility": 0.05,
        "metadata": {},
    }

    provenance = _build_provenance()
    assert registry.register_champion(genome, dict(fitness_report), provenance=provenance)

    if status != "evolved":
        assert registry.update_strategy_status(genome.id, status)

    stored = registry.get_strategy(genome.id)
    assert stored is not None
    assert stored["seed_source"] == "catalogue"
    assert stored["catalogue_name"] == provenance["catalogue"]["name"]
    assert stored["catalogue_entry_id"] == provenance["entry"]["id"]
    assert stored["status"] == status
    report_metadata = stored["fitness_report"].get("metadata", {})
    assert report_metadata["catalogue_provenance"]["catalogue"]["version"] == "2025.09"

    summary = registry.get_registry_summary()
    assert summary["total_strategies"] == 1
    if status == "approved":
        assert summary["approved_count"] == 1
    else:
        assert summary["evolved_count"] == 1
    assert summary["catalogue_seeded"] == 1
    assert summary["catalogue_entry_count"] == 1
    assert summary["catalogue_missing_provenance"] == 0
    assert summary["seed_source_counts"].get("catalogue") == 1


def test_strategy_registry_handles_malformed_payload(tmp_path, caplog: pytest.LogCaptureFixture) -> None:
    registry = StrategyRegistry(db_path=str(tmp_path / "registry.db"))
    with registry.conn:
        registry.conn.execute(
            """
            INSERT INTO strategies (
                genome_id,
                created_at,
                dna,
                fitness_report,
                status,
                catalogue_metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("broken", datetime.now(), "{", "{", "evolved", "{"),
        )

    with caplog.at_level(logging.ERROR):
        assert registry.get_strategy("broken") is None
    assert any(
        "Corrupted JSON payload retrieving strategy" in record.message for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert registry.get_champion_strategies(limit=1) == []
    assert any(
        "Corrupted JSON retrieving champion strategies" in record.message
        for record in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        summary = registry.get_registry_summary()
    assert "error" in summary
    assert any(
        "Corrupted JSON while building registry summary" in record.message
        for record in caplog.records
    )
