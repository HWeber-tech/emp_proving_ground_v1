from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable, Optional, cast

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
    """Consolidated evolution engine surface."""

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        population_manager: PopulationManagerProtocol | None = None,
    ) -> None:
        self.config = config or EvolutionConfig()
        self._population_manager: PopulationManagerProtocol = population_manager or PopulationManagerImpl(
            population_size=self.config.population_size
        )
        self._initialized = False
        self._genome_counter = 0

    def evolve(
        self, genome_factory: Optional[Callable[[], DecisionGenome]] = None
    ) -> EvolutionSummary:
        """Run a minimal evolution cycle and return a summary of the results."""

        factory = genome_factory or self._default_genome_factory

        if not self._initialized:
            self._population_manager.initialize_population(factory)
            self._initialized = True

        if not self._population_manager.get_population():
            self._population_manager.initialize_population(factory)

        elite_count = max(1, self.config.elite_count)
        elites = list(self._population_manager.get_best_genomes(elite_count))

        if elites:
            new_population = list(elites)
            while len(new_population) < self.config.population_size:
                new_population.extend(elites)
            new_population = new_population[: self.config.population_size]
            self._population_manager.update_population(new_population)
        else:
            self._population_manager.initialize_population(factory)

        self._population_manager.advance_generation()
        stats = self._population_manager.get_population_statistics()

        return EvolutionSummary(
            generation=int(stats.get("generation", 0)),
            population_size=int(stats.get("population_size", 0)),
            best_fitness=float(stats.get("best_fitness", 0.0) or 0.0),
            average_fitness=float(stats.get("average_fitness", 0.0) or 0.0),
            elite_count=len(elites),
            timestamp=time(),
        )

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
