"""Evolution cycle orchestrator bridging the encyclopedia's governance loop."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
)

from src.core.event_bus import Event, EventBus, get_global_bus
from src.core.evolution.engine import EvolutionEngine, EvolutionSummary
from src.core.interfaces import DecisionGenome
from src.evolution.catalogue_telemetry import (
    EvolutionCatalogueSnapshot,
    build_catalogue_snapshot,
)
from src.evolution.lineage_telemetry import (
    EvolutionLineageSnapshot,
    build_lineage_snapshot,
)

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from src.governance.strategy_registry import StrategyRegistry

__all__ = [
    "FitnessReport",
    "EvaluationRecord",
    "ChampionRecord",
    "EvolutionCycleResult",
    "EvolutionCycleOrchestrator",
    "SupportsChampionRegistry",
]

FitnessCallback = Callable[[DecisionGenome], Awaitable[Any] | Any]


class SupportsChampionRegistry(Protocol):
    """Protocol for registries capable of storing champion genomes."""

    def register_champion(
        self,
        genome: DecisionGenome,
        fitness_report: Mapping[str, Any],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> bool: ...


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class FitnessReport:
    """Normalized view of a genome's fitness evaluation."""

    fitness_score: float
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "fitness_score": float(self.fitness_score),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "total_return": float(self.total_return),
            "volatility": float(self.volatility),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def for_registry(self) -> dict[str, Any]:
        payload = self.as_payload()
        payload.setdefault("fitness_score", float(self.fitness_score))
        return payload


@dataclass(slots=True)
class EvaluationRecord:
    """Evaluation artefact stored for reporting and tests."""

    genome_id: str
    fitness: float
    report: FitnessReport

    def as_payload(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "fitness": float(self.fitness),
            "report": self.report.as_payload(),
        }


@dataclass(slots=True)
class ChampionRecord:
    """Champion metadata tracked across generations."""

    genome_id: str
    fitness: float
    report: FitnessReport
    registered: bool
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    species: str | None = None
    mutation_history: tuple[str, ...] = field(default_factory=tuple)

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "genome_id": self.genome_id,
            "fitness": float(self.fitness),
            "registered": bool(self.registered),
        }
        payload.update(self.report.as_payload())
        payload["parent_ids"] = list(self.parent_ids)
        payload["species"] = self.species
        payload["mutation_history"] = list(self.mutation_history)
        return payload


@dataclass(slots=True)
class EvolutionCycleResult:
    """Return type for orchestrated evolution runs."""

    summary: EvolutionSummary
    evaluations: list[EvaluationRecord]
    champion: ChampionRecord | None

    def as_payload(self) -> dict[str, Any]:
        return {
            "summary": asdict(self.summary),
            "evaluations": [record.as_payload() for record in self.evaluations],
            "champion": self.champion.as_payload() if self.champion else None,
        }


class EvolutionCycleOrchestrator:
    """Coordinates evaluation, selection, and governance registration."""

    def __init__(
        self,
        evolution_engine: EvolutionEngine,
        evaluation_callback: FitnessCallback,
        *,
        strategy_registry: SupportsChampionRegistry | None = None,
        genome_factory: Callable[[], DecisionGenome] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._engine = evolution_engine
        self._callback = evaluation_callback
        self._registry = strategy_registry
        self._genome_factory = genome_factory
        self._event_bus = event_bus
        self.telemetry: dict[str, Any] = {
            "total_generations": 0,
            "best_fitness": float("-inf"),
            "champion": None,
            "last_summary": None,
        }
        self._best_champion: ChampionRecord | None = None
        self._latest_catalogue_snapshot: EvolutionCatalogueSnapshot | None = None
        self._last_catalogue_fingerprint: tuple[object, ...] | None = None
        self._latest_lineage_snapshot: EvolutionLineageSnapshot | None = None
        self._last_lineage_fingerprint: tuple[object, ...] | None = None

    async def run_cycle(self) -> EvolutionCycleResult:
        """Evaluate the current population, evolve, and optionally register a champion."""

        population = self._engine.ensure_population(self._genome_factory)
        evaluations = await self._evaluate_population(population)
        champion = self._promote_champion(population, evaluations)
        summary = self._engine.evolve(self._genome_factory)
        self._refresh_catalogue_snapshot()
        self._refresh_lineage_snapshot(champion, summary)

        self.telemetry["total_generations"] += 1
        self.telemetry["last_summary"] = asdict(summary)
        if champion is not None:
            self.telemetry["best_fitness"] = max(
                _as_float(self.telemetry.get("best_fitness", 0.0)), champion.fitness
            )
            self.telemetry["champion"] = champion.as_payload()
            self._best_champion = champion
        else:
            self.telemetry["champion"] = None

        if self._latest_lineage_snapshot is not None:
            self.telemetry["lineage"] = self._latest_lineage_snapshot.as_dict()
        elif "lineage" in self.telemetry:
            self.telemetry.pop("lineage", None)

        if self._latest_catalogue_snapshot is not None:
            self.telemetry["catalogue"] = self._latest_catalogue_snapshot.as_dict()
        elif "catalogue" in self.telemetry:
            self.telemetry.pop("catalogue", None)

        return EvolutionCycleResult(summary=summary, evaluations=evaluations, champion=champion)

    async def _evaluate_population(
        self, population: Sequence[DecisionGenome]
    ) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        for genome in population:
            report = await self._evaluate_genome(genome)
            record = EvaluationRecord(
                genome_id=str(getattr(genome, "id", "unknown")),
                fitness=float(report.fitness_score),
                report=report,
            )
            records.append(record)
        return records

    async def _evaluate_genome(self, genome: DecisionGenome) -> FitnessReport:
        try:
            result = self._callback(genome)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=30.0)
        except Exception as exc:  # pragma: no cover - defensive guard
            report = FitnessReport(fitness_score=0.0, metadata={"error": str(exc)})
        else:
            report = self._normalise_report(result)

        self._apply_report_to_genome(genome, report)
        return report

    def _apply_report_to_genome(self, genome: DecisionGenome, report: FitnessReport) -> None:
        try:
            setattr(genome, "fitness", float(report.fitness_score))
        except Exception:  # pragma: no cover - legacy genome compatibility
            pass

        metrics = report.as_payload()
        metrics.pop("metadata", None)
        try:
            existing = getattr(genome, "performance_metrics", {}) or {}
            merged = dict(existing)
            merged.update(metrics)
            setattr(genome, "performance_metrics", merged)
        except Exception:  # pragma: no cover - optional attribute
            pass

        if report.metadata:
            try:
                meta = getattr(genome, "metadata", {}) or {}
                if isinstance(meta, Mapping):
                    updated = dict(meta)
                else:
                    updated = {}
                updated.update(report.metadata)
                setattr(genome, "metadata", updated)
            except Exception:  # pragma: no cover - optional attribute
                pass

    def _promote_champion(
        self,
        population: Sequence[DecisionGenome],
        evaluations: Iterable[EvaluationRecord],
    ) -> ChampionRecord | None:
        try:
            best_record = max(evaluations, key=lambda r: r.fitness)
        except ValueError:
            return None

        genome = next(
            (g for g in population if str(getattr(g, "id", "")) == best_record.genome_id),
            None,
        )
        if genome is None:
            return None

        registered = False
        if self._registry is not None:
            try:
                provenance = self._build_catalogue_provenance(genome)
                registry_payload = best_record.report.for_registry()
                if provenance:
                    metadata = registry_payload.setdefault("metadata", {})
                    if isinstance(metadata, MutableMapping):
                        metadata.setdefault("catalogue_provenance", dict(provenance))
                else:
                    provenance = None
                registered = self._registry.register_champion(
                    genome,
                    registry_payload,
                    provenance=provenance,
                )
            except Exception:  # pragma: no cover - registry failure guard
                registered = False

        champion = ChampionRecord(
            genome_id=best_record.genome_id,
            fitness=best_record.fitness,
            report=best_record.report,
            registered=registered,
            parent_ids=tuple(
                str(pid)
                for pid in getattr(genome, "parent_ids", [])
                if isinstance(pid, str) and pid
            ),
            species=getattr(genome, "species_type", None),
            mutation_history=tuple(
                str(step)
                for step in getattr(genome, "mutation_history", [])
                if isinstance(step, str) and step
            ),
        )
        self._best_champion = champion
        return champion

    def _build_catalogue_provenance(self, genome: DecisionGenome) -> Mapping[str, Any] | None:
        try:
            stats = self._engine.get_population_statistics()
        except Exception:  # pragma: no cover - defensive guard
            return None

        snapshot = build_catalogue_snapshot(stats)
        if snapshot is None:
            return None

        base_identifier = str(getattr(genome, "id", ""))
        if "::" in base_identifier:
            base_identifier = base_identifier.split("::", 1)[0]

        entry_payload: Mapping[str, Any] | None = None
        for entry in snapshot.entries:
            if entry.identifier == base_identifier:
                entry_payload = entry.as_dict()
                break

        provenance: dict[str, Any] = {
            "seed_source": snapshot.seed_source,
            "generation": snapshot.generation,
            "population_size": snapshot.population_size,
            "catalogue": {
                "name": snapshot.catalogue_name,
                "version": snapshot.catalogue_version,
                "size": snapshot.catalogue_size,
                "species": dict(snapshot.catalogue_species),
                "seeded_at": snapshot.seeded_at,
                "source_notes": list(snapshot.source_notes),
            },
        }
        if entry_payload is not None:
            provenance["entry"] = dict(entry_payload)
        return provenance

    def _normalise_report(self, payload: Any) -> FitnessReport:
        if isinstance(payload, FitnessReport):
            return payload

        data: Mapping[str, Any]
        if is_dataclass(payload):
            data = asdict(payload)
        elif isinstance(payload, Mapping):
            data = payload
        else:
            data = {}

        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {"raw_metadata": metadata}

        return FitnessReport(
            fitness_score=_as_float(
                data.get("fitness_score", data.get("fitness", 0.0)), default=0.0
            ),
            max_drawdown=_as_float(data.get("max_drawdown", 0.0)),
            sharpe_ratio=_as_float(data.get("sharpe_ratio", data.get("sharpe", 0.0))),
            total_return=_as_float(data.get("total_return", data.get("return", 0.0))),
            volatility=_as_float(data.get("volatility", data.get("sigma", 0.0))),
            metadata=dict(metadata),
        )

    @property
    def champion(self) -> ChampionRecord | None:
        """Return the best champion observed so far."""

        return self._best_champion

    @property
    def catalogue_snapshot(self) -> EvolutionCatalogueSnapshot | None:
        """Return the latest catalogue snapshot recorded during evolution."""

        return self._latest_catalogue_snapshot

    @property
    def lineage_snapshot(self) -> EvolutionLineageSnapshot | None:
        """Expose the latest lineage snapshot for monitoring surfaces."""

        return self._latest_lineage_snapshot

    @property
    def population_statistics(self) -> Mapping[str, object]:
        """Expose population statistics for monitoring surfaces."""

        return self._engine.get_population_statistics()

    def _refresh_catalogue_snapshot(self) -> None:
        stats = self._engine.get_population_statistics()
        snapshot = build_catalogue_snapshot(stats)
        self._latest_catalogue_snapshot = snapshot
        if snapshot is not None:
            fingerprint = snapshot.fingerprint()
            if fingerprint != self._last_catalogue_fingerprint:
                self._last_catalogue_fingerprint = fingerprint
                self._publish_catalogue_snapshot(snapshot)

    def _publish_catalogue_snapshot(
        self, snapshot: EvolutionCatalogueSnapshot, *, max_entries: int = 8
    ) -> None:
        payload = snapshot.as_dict(max_entries=max_entries)
        event = Event(
            type="telemetry.evolution.catalogue",
            payload=payload,
            source="evolution_orchestrator",
        )

        bus = self._event_bus
        if bus is not None:
            publish_from_sync = getattr(bus, "publish_from_sync", None)
            is_running = getattr(bus, "is_running", lambda: False)
            if callable(publish_from_sync) and callable(is_running) and is_running():
                try:
                    publish_from_sync(event)
                    return
                except Exception:
                    pass

        try:
            topic_bus = get_global_bus()
            topic_bus.publish_sync(event.type, payload, source=event.source)
        except Exception:
            # Diagnostics only – avoid crashing evolution cycle when telemetry fails.
            pass

    def _refresh_lineage_snapshot(
        self, champion: ChampionRecord | None, summary: EvolutionSummary
    ) -> None:
        stats = self._engine.get_population_statistics()
        snapshot = build_lineage_snapshot(stats, champion, summary=summary)
        self._latest_lineage_snapshot = snapshot
        if snapshot is None:
            return

        fingerprint = snapshot.fingerprint()
        if fingerprint != self._last_lineage_fingerprint:
            self._last_lineage_fingerprint = fingerprint
            self._publish_lineage_snapshot(snapshot)

    def _publish_lineage_snapshot(
        self, snapshot: EvolutionLineageSnapshot, *, max_parents: int = 8
    ) -> None:
        payload = snapshot.as_dict(max_parents=max_parents)
        event = Event(
            type="telemetry.evolution.lineage",
            payload=payload,
            source="evolution_orchestrator",
        )

        bus = self._event_bus
        if bus is not None:
            publish_from_sync = getattr(bus, "publish_from_sync", None)
            is_running = getattr(bus, "is_running", lambda: False)
            if callable(publish_from_sync) and callable(is_running) and is_running():
                try:
                    publish_from_sync(event)
                    return
                except Exception:
                    pass

        try:
            topic_bus = get_global_bus()
            topic_bus.publish_sync(event.type, payload, source=event.source)
        except Exception:
            # Diagnostics only – avoid crashing evolution cycle when telemetry fails.
            pass
