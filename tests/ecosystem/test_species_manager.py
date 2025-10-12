from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ecosystem.species.species_manager import SpeciesManager


@pytest.mark.asyncio()
async def test_species_manager_evolve_specialist_records_history() -> None:
    manager = SpeciesManager()
    niche = SimpleNamespace(regime_type="storm", opportunity_score=0.85)

    specialist = await manager.evolve_specialist(
        niche=niche,
        base_population=[object(), object()],
        specialization_pressure=0.75,
    )

    assert specialist.predator_id == "predator-1"
    assert specialist.predator_type == "storm"
    assert 0.0 <= specialist.performance_metrics["adaptation_score"] <= 1.0
    metrics = manager.get_population_metrics()
    assert metrics["total_evolved"] == 1
    assert metrics["last_evolved"]["predator_type"] == "storm"
    history = manager.list_history()
    assert len(history) == 1
    assert history[0]["population_size"] == 2


@pytest.mark.asyncio()
async def test_species_manager_handles_empty_history() -> None:
    manager = SpeciesManager()
    metrics = manager.get_population_metrics()
    assert metrics["total_evolved"] == 0
    assert metrics["last_evolved"] is None
    assert metrics["average_pressure"] == 0.0
    assert manager.list_history() == []

