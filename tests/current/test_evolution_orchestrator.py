from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.governance.strategy_registry import StrategyRegistry
from src.orchestration.evolution_cycle import EvolutionCycleOrchestrator


class RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event) -> int:  # pragma: no cover - trivial
        self.events.append(event)
        return 1


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
        EvolutionConfig(
            population_size=6,
            elite_count=2,
            crossover_rate=0.55,
            mutation_rate=0.15,
            use_catalogue=True,
        )
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

    bus = RecordingBus()
    orchestrator = EvolutionCycleOrchestrator(
        engine,
        evaluator,
        strategy_registry=registry,
        event_bus=bus,
        adaptive_runs_enabled=True,
    )

    result = await orchestrator.run_cycle()

    assert len(result.evaluations) == 6
    assert all(record.fitness >= 0.0 for record in result.evaluations)

    decision = orchestrator.adaptive_runs_decision
    assert decision is not None
    assert decision.enabled is True
    assert decision.source == "override"
    assert decision.reason == "override_enabled"
    assert orchestrator.telemetry["adaptive_runs"]["source"] == "override"

    champion = result.champion
    assert champion is not None
    assert champion.fitness == max(record.fitness for record in result.evaluations)
    assert champion.registered is True

    stored = registry.get_strategy(champion.genome_id)
    assert stored is not None
    assert pytest.approx(float(stored["fitness_score"])) == pytest.approx(champion.fitness)
    assert stored["seed_source"] == "catalogue"
    assert stored["catalogue_name"]
    assert stored["catalogue_entry_id"]
    provenance = stored["fitness_report"].get("metadata", {}).get("catalogue_provenance")
    assert provenance
    assert provenance["catalogue"]["name"] == stored["catalogue_name"]
    assert provenance.get("entry", {}).get("id") == stored["catalogue_entry_id"]

    telemetry = orchestrator.telemetry
    assert telemetry["total_generations"] == 1
    assert telemetry["champion"]["genome_id"] == champion.genome_id
    assert telemetry["champion"]["parent_ids"] == list(champion.parent_ids)
    assert telemetry["champion"]["species"] == champion.species
    assert telemetry["champion"]["mutation_history"] == list(champion.mutation_history)
    assert orchestrator.population_statistics["population_size"] == 6

    lineage = telemetry["lineage"]
    assert lineage["champion"]["id"] == champion.genome_id
    assert lineage["champion"]["metadata"]["evaluated"] is True
    assert lineage["population"]["seed_source"] in {"factory", "catalogue", "realistic_sampler"}
    seed_metadata = lineage["population"].get("seed_metadata") or {}
    if seed_metadata:
        assert seed_metadata["seed_names"]
        assert seed_metadata["seed_templates"]

    snapshot = orchestrator.lineage_snapshot
    assert snapshot is not None
    assert snapshot.champion_id == champion.genome_id
    assert snapshot.to_markdown().startswith("### Evolution lineage")
    if snapshot.seed_metadata:
        assert snapshot.seed_metadata.get("seed_names")

    lineage_events = [event for event in bus.events if event.type == "telemetry.evolution.lineage"]
    assert lineage_events, "expected lineage telemetry event"
    assert lineage_events[-1].payload["champion"]["id"] == champion.genome_id
    if seed_metadata:
        assert lineage_events[-1].payload["population"].get("seed_metadata")

    summary = registry.get_registry_summary()
    assert summary["catalogue_seeded"] >= 1
    assert summary["catalogue_entry_count"] >= 1


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

    orchestrator = EvolutionCycleOrchestrator(engine, evaluator, adaptive_runs_enabled=True)

    first = await orchestrator.run_cycle()
    second = await orchestrator.run_cycle()

    assert first.champion is not None
    assert second.champion is not None
    assert orchestrator.champion is second.champion
    assert orchestrator.telemetry["total_generations"] == 2
    assert orchestrator.telemetry["champion"]["registered"] is False
    assert orchestrator.telemetry["lineage"]["champion"]["id"] == second.champion.genome_id
    assert orchestrator.lineage_snapshot is not None

    # Ensure metadata normalization handled the None metadata gracefully
    assert all("fitness_score" in record.report.as_payload() for record in second.evaluations)
    assert orchestrator.population_statistics["population_size"] == 3


@pytest.mark.asyncio
async def test_catalogue_snapshot_emitted_when_catalogue_active():
    engine = EvolutionEngine(
        EvolutionConfig(
            population_size=4,
            elite_count=1,
            crossover_rate=0.5,
            mutation_rate=0.1,
            use_catalogue=True,
        )
    )
    engine._rng.seed(13)  # type: ignore[attr-defined]

    def evaluator(genome):
        return {
            "fitness_score": _score_parameters(genome),
            "sharpe_ratio": 1.0,
        }

    bus = RecordingBus()
    orchestrator = EvolutionCycleOrchestrator(
        engine, evaluator, event_bus=bus, adaptive_runs_enabled=True
    )

    result = await orchestrator.run_cycle()
    assert result.champion is not None

    snapshot = orchestrator.catalogue_snapshot
    assert snapshot is not None
    assert snapshot.seed_source == "catalogue"
    assert snapshot.population_size == 4

    assert bus.events, "expected telemetry event to be published"
    lineage_events = [event for event in bus.events if event.type == "telemetry.evolution.lineage"]
    assert lineage_events, "expected lineage telemetry event"
    assert lineage_events[-1].payload["champion"]["id"] == result.champion.genome_id

    catalogue_events = [
        event for event in bus.events if event.type == "telemetry.evolution.catalogue"
    ]
    assert catalogue_events, "expected catalogue telemetry event"
    last_catalogue = catalogue_events[-1]
    assert last_catalogue.payload["catalogue"]["name"] == snapshot.catalogue_name
    assert last_catalogue.payload["generation"] == snapshot.generation


@pytest.mark.asyncio
async def test_orchestrator_skips_adaptive_runs_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("EVOLUTION_ENABLE_ADAPTIVE_RUNS", raising=False)

    engine = EvolutionEngine(
        EvolutionConfig(
            population_size=3,
            elite_count=1,
            crossover_rate=0.4,
            mutation_rate=0.1,
            use_catalogue=True,
        )
    )
    engine._rng.seed(17)  # type: ignore[attr-defined]

    registry = StrategyRegistry(db_path=str(tmp_path / "governance-disabled.db"))

    async def evaluator(genome):
        return {"fitness_score": _score_parameters(genome), "sharpe_ratio": 0.9}

    orchestrator = EvolutionCycleOrchestrator(
        engine,
        evaluator,
        strategy_registry=registry,
        adaptive_runs_enabled=False,
    )

    first = await orchestrator.run_cycle()
    assert first.summary.generation == 0
    assert orchestrator.telemetry["adaptive_runs_enabled"] is False
    assert orchestrator.telemetry["adaptive_runs"]["reason"] == "override_disabled"
    assert orchestrator.telemetry["total_generations"] == 0

    decision = orchestrator.adaptive_runs_decision
    assert decision is not None
    assert decision.enabled is False
    assert decision.reason == "override_disabled"

    champion = first.champion
    assert champion is not None
    assert champion.registered is False
    assert registry.get_strategy(champion.genome_id) is None
    seed_metadata = orchestrator.telemetry["lineage"]["population"].get("seed_metadata")
    if seed_metadata:
        assert seed_metadata["seed_names"]

    second = await orchestrator.run_cycle()
    assert second.summary.generation == 0
    assert orchestrator.telemetry["total_generations"] == 0

    stats = engine.get_population_statistics()
    assert stats["generation"] == 0


@pytest.mark.asyncio
async def test_seed_metadata_present_when_realistic_seeder_used(monkeypatch):
    monkeypatch.delenv("EVOLUTION_ENABLE_ADAPTIVE_RUNS", raising=False)

    engine = EvolutionEngine(
        EvolutionConfig(
            population_size=4,
            elite_count=1,
            crossover_rate=0.4,
            mutation_rate=0.1,
            use_catalogue=False,
        )
    )
    engine._rng.seed(29)  # type: ignore[attr-defined]

    async def evaluator(genome):
        return {"fitness_score": _score_parameters(genome), "sharpe_ratio": 1.1}

    orchestrator = EvolutionCycleOrchestrator(
        engine,
        evaluator,
        adaptive_runs_enabled=False,
    )

    result = await orchestrator.run_cycle()
    assert result.champion is not None

    seed_metadata = orchestrator.telemetry["lineage"]["population"].get("seed_metadata")
    assert seed_metadata, "expected seed metadata from realistic sampler"
    assert seed_metadata["seed_names"]
    assert seed_metadata["seed_templates"]

    snapshot = orchestrator.lineage_snapshot
    assert snapshot is not None
    assert snapshot.seed_metadata
    assert snapshot.seed_metadata.get("seed_names")


@pytest.mark.asyncio
async def test_orchestrator_uses_environment_flag(monkeypatch):
    monkeypatch.setenv("EVOLUTION_ENABLE_ADAPTIVE_RUNS", "true")

    engine = EvolutionEngine(
        EvolutionConfig(population_size=3, elite_count=1, crossover_rate=0.4, mutation_rate=0.1)
    )
    engine._rng.seed(31)  # type: ignore[attr-defined]

    async def evaluator(genome):
        return {"fitness_score": _score_parameters(genome), "sharpe_ratio": 0.8}

    orchestrator = EvolutionCycleOrchestrator(engine, evaluator)

    result = await orchestrator.run_cycle()
    assert result.champion is not None

    decision = orchestrator.adaptive_runs_decision
    assert decision is not None
    assert decision.source == "environment"
    assert decision.reason == "flag_enabled"
    assert orchestrator.telemetry["adaptive_runs"]["reason"] == "flag_enabled"
