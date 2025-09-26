from __future__ import annotations

import random
from dataclasses import dataclass
from time import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, cast

from src.core.genome import get_genome_provider
from src.core.interfaces import DecisionGenome, PopulationManager as PopulationManagerProtocol
from src.core.population_manager import PopulationManager as PopulationManagerImpl


@dataclass
class EvolutionConfig:
    population_size: int = 100
    elite_count: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    max_generations: int = 100
    use_catalogue: bool | None = None


@dataclass
class EvolutionSummary:
    """Lightweight summary describing the outcome of an evolution step."""

    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    elite_count: int
    timestamp: float


class EvolutionEngine:
    """Consolidated evolution engine surface with crossover and mutation."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        population_manager: PopulationManagerProtocol | None = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self._population_manager: PopulationManagerProtocol = (
            population_manager
            or PopulationManagerImpl(
                population_size=self.config.population_size,
                use_catalogue=self.config.use_catalogue,
            )
        )
        self._initialized = False
        self._genome_counter = 0
        self._rng = random.Random()

    def ensure_population(
        self, genome_factory: Optional[Callable[[], DecisionGenome]] = None
    ) -> Sequence[DecisionGenome]:
        """Ensure a population exists and return a snapshot of it."""

        factory = genome_factory or self._default_genome_factory

        if not self._initialized:
            self._population_manager.initialize_population(factory)
            self._initialized = True

        population = self._population_manager.get_population()
        if not population:
            seeds = [factory() for _ in range(self.config.population_size)]
            self._population_manager.update_population(seeds)
            population = self._population_manager.get_population()

        return list(population)

    def get_population(self) -> Sequence[DecisionGenome]:
        """Expose the current population for orchestration layers."""

        return list(self._population_manager.get_population())

    def get_population_statistics(self) -> Mapping[str, object]:
        """Return the latest population statistics."""

        return dict(self._population_manager.get_population_statistics())

    def evolve(
        self, genome_factory: Optional[Callable[[], DecisionGenome]] = None
    ) -> EvolutionSummary:
        """Run a full evolution cycle incorporating selection, crossover and mutation."""

        factory = genome_factory or self._default_genome_factory

        population = self.ensure_population(factory)

        if not population:
            stats = self._population_manager.get_population_statistics()
            return EvolutionSummary(
                generation=_to_int(stats.get("generation", 0)),
                population_size=0,
                best_fitness=0.0,
                average_fitness=0.0,
                elite_count=0,
                timestamp=time(),
            )

        stats_before = self._population_manager.get_population_statistics()
        current_generation = _to_int(stats_before.get("generation", 0))

        elite_count = max(1, min(self.config.elite_count, len(population)))
        elites = list(self._population_manager.get_best_genomes(elite_count))
        if not elites:
            elites = [population[0]]

        new_population: list[DecisionGenome] = list(elites)
        while len(new_population) < self.config.population_size:
            parent1 = self._select_parent(population)
            parent2 = self._select_parent(population) if len(population) > 1 else parent1
            if parent2 is parent1 and len(population) > 1:
                parent2 = self._select_alternate_parent(population, parent1)

            params_a = self._extract_parameters(parent1)
            params_b = self._extract_parameters(parent2)

            if parent1 is parent2 or self._rng.random() >= self.config.crossover_rate:
                child_params = dict(params_a)
            else:
                child_params = self._crossover_parameters(params_a, params_b)

            child_params = self._mutate_parameters(child_params, self.config.mutation_rate)

            species = (
                getattr(parent1, "species_type", None)
                or getattr(parent2, "species_type", None)
                or "core_strategy"
            )
            child = self._spawn_genome(
                parameters=child_params,
                species=species,
                parents=[parent for parent in (parent1, parent2) if parent is not None],
                generation=current_generation + 1,
            )

            new_population.append(child)

            # Periodically inject a fresh random genome to maintain diversity
            if self._rng.random() < 0.05 and len(new_population) < self.config.population_size:
                seed_child = self._spawn_genome(
                    parameters=self._extract_parameters(factory()),
                    species="core_strategy",
                    parents=[],
                    generation=current_generation + 1,
                )
                new_population.append(seed_child)

        self._population_manager.update_population(new_population[: self.config.population_size])
        self._population_manager.advance_generation()
        stats = self._population_manager.get_population_statistics()

        return EvolutionSummary(
            generation=_to_int(stats.get("generation", 0)),
            population_size=_to_int(stats.get("population_size", 0)),
            best_fitness=_to_float(stats.get("best_fitness", 0.0), default=0.0) or 0.0,
            average_fitness=_to_float(stats.get("average_fitness", 0.0), default=0.0) or 0.0,
            elite_count=len(elites),
            timestamp=time(),
        )

    def _select_parent(self, population: Sequence[DecisionGenome]) -> DecisionGenome:
        if len(population) == 1:
            return population[0]
        tournament = self._rng.sample(population, k=min(3, len(population)))
        return max(tournament, key=lambda g: float(getattr(g, "fitness", 0.0) or 0.0))

    def _select_alternate_parent(
        self, population: Sequence[DecisionGenome], exclude: DecisionGenome
    ) -> DecisionGenome:
        candidates = [g for g in population if g is not exclude]
        if not candidates:
            return exclude
        return self._select_parent(candidates)

    def _extract_parameters(self, genome: DecisionGenome) -> Dict[str, float]:
        params = getattr(genome, "parameters", {})
        result: Dict[str, float] = {}
        if isinstance(params, Mapping):
            iterator: Iterable[tuple[object, object]] = params.items()
        elif hasattr(params, "__dict__"):
            iterator = vars(params).items()
        else:
            iterator = ()
        for key, value in iterator:
            coerced = _to_float(value, default=None)
            if coerced is None:
                continue
            result[str(key)] = coerced
        return result

    def _crossover_parameters(
        self, params_a: Mapping[str, float], params_b: Mapping[str, float]
    ) -> Dict[str, float]:
        child: Dict[str, float] = {}
        keys = set(params_a.keys()) | set(params_b.keys())
        for key in keys:
            a = float(params_a.get(key, params_b.get(key, 0.0)))
            b = float(params_b.get(key, params_a.get(key, 0.0)))
            weight = self._rng.random()
            child[key] = weight * a + (1.0 - weight) * b
        return child

    def _mutate_parameters(
        self, parameters: Mapping[str, float], mutation_rate: float
    ) -> Dict[str, float]:
        mutated = dict(parameters)
        for key, value in list(mutated.items()):
            if self._rng.random() < mutation_rate:
                scale = abs(value) if value else 1.0
                mutated[key] = float(value + self._rng.gauss(0.0, 0.15 * scale))
        return mutated

    def _spawn_genome(
        self,
        parameters: Mapping[str, float],
        species: str,
        parents: Sequence[DecisionGenome],
        generation: int,
    ) -> DecisionGenome:
        provider = get_genome_provider()
        self._genome_counter += 1
        identifier = f"core-evo-{self._genome_counter:05d}"
        genome = provider.new_genome(
            id=identifier,
            parameters=dict(parameters),
            generation=generation,
            species_type=species,
        )
        if not isinstance(genome, DecisionGenome):
            genome = cast(DecisionGenome, provider.from_legacy(genome))

        parent_ids = [
            getattr(parent, "id", "") for parent in parents if getattr(parent, "id", None)
        ]
        genome = self._apply_parent_metadata(genome, parent_ids, generation)
        self._normalize_genome(genome)
        try:
            setattr(genome, "fitness", None)
        except Exception:
            pass
        return genome

    def _apply_parent_metadata(
        self, genome: DecisionGenome, parent_ids: Sequence[str], generation: int
    ) -> DecisionGenome:
        try:
            if hasattr(genome, "with_updated"):
                genome = cast(
                    DecisionGenome,
                    genome.with_updated(parent_ids=list(parent_ids), generation=int(generation)),
                )
            else:
                if parent_ids and hasattr(genome, "parent_ids"):
                    setattr(genome, "parent_ids", list(parent_ids))
                if hasattr(genome, "generation"):
                    setattr(genome, "generation", int(generation))
        except Exception:
            pass
        return genome

    def _normalize_genome(self, genome: DecisionGenome) -> None:
        try:
            normalizer = getattr(genome, "_normalize_weights", None)
            if callable(normalizer):
                normalizer()
        except Exception:
            pass

    def _default_genome_factory(self) -> DecisionGenome:
        provider = get_genome_provider()
        self._genome_counter += 1
        identifier = f"core-evo-{self._genome_counter:05d}"
        parameters = {
            "risk_tolerance": 0.5,
            "momentum_window": float(10 + (self._genome_counter % 15)),
            "mean_reversion": 0.2,
        }
        genome = provider.new_genome(
            id=identifier,
            parameters=parameters,
            generation=0,
            species_type="core_strategy",
        )
        if isinstance(genome, DecisionGenome):
            return genome
        return cast(DecisionGenome, provider.from_legacy(genome))
def _to_int(value: object, *, default: int = 0) -> int:
    """Return a safe ``int`` conversion for telemetry payload values."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _to_float(value: object, *, default: float | None = 0.0) -> float | None:
    """Return a safe ``float`` conversion for telemetry payload values."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
