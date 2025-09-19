from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.governance.strategy_registry import StrategyRegistry
from src.orchestration.evolution_cycle import EvolutionCycleOrchestrator


def _score_parameters(genome: object) -> float:
    params = getattr(genome, "parameters", {}) or {}
    try:
        values = [float(value) for value in params.values()]
    except Exception:  # pragma: no cover - defensive guard
        values = []
    if not values:
        return 0.0
    return sum(values) / float(len(values))


@pytest.mark.asyncio
async def test_orchestrator_registers_champion_and_updates_registry(tmp_path):
    engine = EvolutionEngine(
        EvolutionConfig(population_size=6, elite_count=2, crossover_rate=0.55, mutation_rate=0.15)
    )
    engine._rng.seed(7)  # type: ignore[attr-defined]

    registry = StrategyRegistry(db_path=str(tmp_path / "governance.db"))

    async def evaluator(genome):
        score = _score_parameters(genome)
        return {
            "fitness_score": score,
            "max_drawdown": 0.05 + 0.001 * len(getattr(genome, "mutation_history", [])),
            "sharpe_ratio": 1.2,
            "total_return": score * 0.25,
            "volatility": 0.15,
            "metadata": {"evaluated": True, "genome": getattr(genome, "id", "")},
        }

    orchestrator = EvolutionCycleOrchestrator(
        engine,
        evaluator,
        strategy_registry=registry,
    )

    result = await orchestrator.run_cycle()

    assert len(result.evaluations) == 6
    assert all(record.fitness >= 0.0 for record in result.evaluations)

    champion = result.champion
    assert champion is not None
    assert champion.fitness == max(record.fitness for record in result.evaluations)
    assert champion.registered is True

    stored = registry.get_strategy(champion.genome_id)
    assert stored is not None
    assert pytest.approx(float(stored["fitness_score"])) == pytest.approx(champion.fitness)

    telemetry = orchestrator.telemetry
    assert telemetry["total_generations"] == 1
    assert telemetry["champion"]["genome_id"] == champion.genome_id
    assert orchestrator.population_statistics["population_size"] == 6


@dataclass
class SimpleReport:
    fitness_score: float
    sharpe_ratio: float
    metadata: dict[str, float] | None = None


@pytest.mark.asyncio
async def test_orchestrator_supports_sync_evaluators_and_dataclass_reports():
    engine = EvolutionEngine(
        EvolutionConfig(population_size=3, elite_count=1, crossover_rate=0.6, mutation_rate=0.2)
    )
    engine._rng.seed(11)  # type: ignore[attr-defined]

    def evaluator(genome):
        base = _score_parameters(genome)
        return SimpleReport(fitness_score=0.1 + base, sharpe_ratio=1.5)

    orchestrator = EvolutionCycleOrchestrator(engine, evaluator)

    first = await orchestrator.run_cycle()
    second = await orchestrator.run_cycle()

    assert first.champion is not None
    assert second.champion is not None
    assert orchestrator.champion is second.champion
    assert orchestrator.telemetry["total_generations"] == 2
    assert orchestrator.telemetry["champion"]["registered"] is False

    # Ensure metadata normalization handled the None metadata gracefully
    assert all("fitness_score" in record.report.as_payload() for record in second.evaluations)
    assert orchestrator.population_statistics["population_size"] == 3
