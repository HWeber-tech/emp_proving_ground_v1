"""
Population Manager Implementation
=================================

Concrete implementation of IPopulationManager for genetic algorithm population management.
Optimized for performance with Redis caching integration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np

from src.core.evolution.seeding import (
    RealisticGenomeSeeder,
    apply_seed_to_genome,
    summarize_seed_metadata,
)
from src.core.genome import GenomeProvider, get_genome_provider
from src.core.performance.market_data_cache import get_global_cache
from src.genome.catalogue import GenomeCatalogue, load_default_catalogue

from .interfaces import DecisionGenome, IPopulationManager

logger = logging.getLogger(__name__)


class PopulationManager(IPopulationManager):
    """High-performance population manager with caching support."""

    def __init__(
        self,
        population_size: int = 100,
        cache_ttl: int = 300,
        *,
        use_catalogue: bool | None = None,
        catalogue: GenomeCatalogue | None = None,
    ):
        """
        Initialize population manager.

        Args:
            population_size: Size of the population
            cache_ttl: Cache TTL in seconds
        """
        self.population_size = population_size
        self.cache_ttl = cache_ttl
        self.population: List[DecisionGenome] = []
        self.generation = 0
        self.cache = get_global_cache()
        self._cache_key_prefix = "population"
        self._catalogue_flag: bool | None = use_catalogue
        self._catalogue: GenomeCatalogue | None = catalogue
        self._catalogue_summary: Dict[str, object] | None = None
        self._catalogue_seeded_at: float | None = None
        self._seed_source: str = "factory"
        self._seed_metadata: Dict[str, object] | None = None
        self._seed_sampler: RealisticGenomeSeeder | None = self._build_seed_sampler()

    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize population with new genomes."""
        logger.info(f"Initializing population with {self.population_size} genomes")
        provider = get_genome_provider()
        self.population = []

        catalogue_seeded = False
        catalogue = self._get_catalogue()
        if catalogue is not None:
            seeded = self._seed_from_catalogue(provider, catalogue)
            if seeded:
                catalogue_seeded = True
                self.population = seeded

        if not catalogue_seeded:
            self.population = [
                cast(DecisionGenome, provider.from_legacy(genome_factory()))
                for _ in range(self.population_size)
            ]
            self._catalogue_summary = None
            self._catalogue_seeded_at = None
            seed_summary = summarize_seed_metadata(self.population)
            self.record_seed_metadata(seed_summary, seed_source="factory")
        self.generation = 0
        self._cache_population_stats()

    def get_population(self) -> List[DecisionGenome]:
        """Get current population."""
        return cast(List[DecisionGenome], self.population.copy())

    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """Get top N genomes by fitness."""
        if not self.population:
            # Generate initial population if empty
            logger.info("Population empty, generating initial population")
            self._generate_initial_population()

        if not self.population:
            logger.warning("Failed to generate initial population")
            return []

        # Sort by fitness (descending), treating None as 0.0
        sorted_population = sorted(
            self.population,
            key=lambda g: (g.fitness or 0.0),
            reverse=True,
        )
        return cast(List[DecisionGenome], sorted_population[: min(count, len(sorted_population))])

    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """Replace current population with new one."""
        logger.info(f"Updating population with {len(new_population)} genomes")
        provider = get_genome_provider()
        self.population = [cast(DecisionGenome, provider.from_legacy(g)) for g in new_population]
        self._cache_population_stats()

    def record_seed_metadata(
        self, metadata: Mapping[str, object] | None, *, seed_source: str | None = None
    ) -> None:
        """Record seed provenance metadata for population statistics."""

        extracted_source: str | None = None
        if isinstance(metadata, Mapping):
            normalised: Dict[str, object] = {}
            for key, value in metadata.items():
                key_text = str(key)
                if key_text == "seed_source" and isinstance(value, str):
                    extracted_source = value
                    continue
                if isinstance(value, Mapping):
                    normalised[key_text] = {
                        str(inner_key): inner_value for inner_key, inner_value in value.items()
                    }
                else:
                    normalised[key_text] = value
            self._seed_metadata = normalised or None
        else:
            self._seed_metadata = None

        resolved_source = seed_source or extracted_source
        if resolved_source:
            self._seed_source = resolved_source

    def get_population_statistics(self) -> dict[str, object]:
        """Get statistics about the current population."""
        if not self.population:
            stats = {
                "generation": self.generation,
                "population_size": 0,
                "average_fitness": 0.0,
                "best_fitness": 0.0,
                "worst_fitness": 0.0,
                "fitness_std": 0.0,
                "species_distribution": {},
                "seed_source": self._seed_source,
            }
            if self._catalogue_summary:
                stats["catalogue"] = dict(self._catalogue_summary)
            if self._seed_metadata:
                stats["seed_metadata"] = dict(self._seed_metadata)
            return stats

        fitness_values = [(g.fitness or 0.0) for g in self.population]

        # Species distribution
        species_count: Dict[str, int] = {}
        for genome in self.population:
            species = genome.species_type or "generic"
            species_count[species] = species_count.get(species, 0) + 1

        stats = {
            "generation": self.generation,
            "population_size": len(self.population),
            "average_fitness": float(np.mean(fitness_values)),
            "best_fitness": float(np.max(fitness_values)) if fitness_values else 0.0,
            "worst_fitness": float(np.min(fitness_values)) if fitness_values else 0.0,
            "fitness_std": float(np.std(fitness_values)) if fitness_values else 0.0,
            "species_distribution": species_count,
            "seed_source": self._seed_source,
        }
        if self._catalogue_summary:
            stats["catalogue"] = dict(self._catalogue_summary)
        if self._seed_metadata:
            stats["seed_metadata"] = dict(self._seed_metadata)
        return stats

    def advance_generation(self) -> None:
        """Increment the generation counter."""
        self.generation += 1
        logger.info(f"Advanced to generation {self.generation}")

    def reset(self) -> None:
        """Reset the population manager to initial state."""
        logger.info("Resetting population manager")
        self.population.clear()
        self.generation = 0
        self._seed_metadata = None

    def _cache_population_stats(self) -> None:
        """Cache population statistics for performance."""
        stats = self.get_population_statistics()
        # Touch legacy view path for compatibility (no-op for performance)
        try:
            provider = get_genome_provider()
            if self.population:
                _ = provider.to_legacy_view(self.population[0])
        except Exception:
            pass
        logger.debug(f"Cached population stats: {stats}")

    def get_genome_by_id(self, genome_id: str) -> Optional[DecisionGenome]:
        """Get a specific genome by ID."""
        for genome in self.population:
            try:
                gid = getattr(genome, "id", None)
                if gid == genome_id:
                    return cast(DecisionGenome, genome)
            except Exception:
                continue
        return None

    def get_species_count(self, species_type: str) -> int:
        """Get count of genomes for a specific species."""
        return sum(1 for g in self.population if (g.species_type or "generic") == species_type)

    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get fitness distribution statistics."""
        if not self.population:
            return {}

        fitness_values = [(g.fitness or 0.0) for g in self.population]
        return {
            "min": float(np.min(fitness_values)),
            "max": float(np.max(fitness_values)),
            "mean": float(np.mean(fitness_values)),
            "median": float(np.median(fitness_values)),
            "std": float(np.std(fitness_values)),
            "percentile_25": float(np.percentile(fitness_values, 25)),
            "percentile_75": float(np.percentile(fitness_values, 75)),
        }

    def _generate_initial_population(self) -> None:
        """Generate initial population of genomes."""
        catalogue = self._get_catalogue()
        provider = get_genome_provider()

        if catalogue is not None:
            seeded = self._seed_from_catalogue(provider, catalogue)
            if seeded:
                self.population.extend(seeded)
                logger.info("Seeded %s genomes from catalogue %s", len(seeded), catalogue.name)
                self.record_seed_metadata(None, seed_source="catalogue")
                return

        seeded, summary = self._seed_with_sampler(provider)
        if seeded:
            self.population.extend(seeded)
            logger.info("Seeded %s genomes using realistic sampler", len(seeded))
            self.record_seed_metadata(summary, seed_source="realistic_sampler")
            return

        try:
            import random

            logger.info(f"Generating initial population of {self.population_size} genomes")

            for i in range(self.population_size):
                params = {
                    "risk_tolerance": random.uniform(0.1, 0.9),
                    "position_size_factor": random.uniform(0.01, 0.1),
                    "stop_loss_factor": random.uniform(0.005, 0.05),
                    "take_profit_factor": random.uniform(0.01, 0.1),
                    "trend_sensitivity": random.uniform(0.1, 1.0),
                    "volatility_threshold": random.uniform(0.001, 0.01),
                    "correlation_threshold": random.uniform(0.3, 0.9),
                    "momentum_window": float(random.randint(5, 50)),
                    "mean_reversion_factor": random.uniform(0.1, 0.8),
                    "market_regime_sensitivity": random.uniform(0.2, 0.8),
                }
                genome = provider.new_genome(
                    id=f"genome_{i:04d}",
                    parameters=params,
                    generation=0,
                    species_type="trading_strategy",
                )
                self.population.append(cast(DecisionGenome, genome))

            logger.info(f"Successfully generated {len(self.population)} genomes")
            self.record_seed_metadata(None, seed_source="factory")

        except Exception as e:
            logger.error(f"Failed to generate initial population: {e}")

    # ------------------------------------------------------------------
    # Catalogue helpers
    # ------------------------------------------------------------------

    def _resolve_catalogue_flag(self) -> bool:
        if self._catalogue_flag is not None:
            return self._catalogue_flag
        raw = os.environ.get("EVOLUTION_USE_CATALOGUE")
        if raw is None:
            self._catalogue_flag = False
            return False
        normalised = raw.strip().lower()
        self._catalogue_flag = normalised in {"1", "true", "yes", "on"}
        return self._catalogue_flag

    def _get_catalogue(self) -> GenomeCatalogue | None:
        if not self._resolve_catalogue_flag():
            return None
        if self._catalogue is None:
            try:
                self._catalogue = load_default_catalogue()
            except Exception as exc:
                logger.error("Failed to load genome catalogue: %s", exc)
                self._catalogue_flag = False
                return None
        return self._catalogue

    def _seed_from_catalogue(
        self, provider: GenomeProvider, catalogue: GenomeCatalogue
    ) -> list[DecisionGenome]:
        try:
            seeds = catalogue.sample(self.population_size)
        except Exception as exc:
            logger.error("Sampling catalogue genomes failed: %s", exc)
            return []

        resolved: list[DecisionGenome] = []
        for genome in seeds:
            try:
                resolved_genome = cast(DecisionGenome, provider.from_legacy(genome))
            except Exception:
                resolved_genome = cast(DecisionGenome, genome)
            resolved.append(resolved_genome)

        summary: dict[str, object] = dict(catalogue.metadata())
        summary["entries"] = catalogue.describe_entries()
        seeded_at = time.time()
        summary["seeded_at"] = seeded_at
        self._catalogue_summary = summary
        self._catalogue_seeded_at = seeded_at

        seed_summary = summarize_seed_metadata(resolved)
        self.record_seed_metadata(seed_summary, seed_source="catalogue")
        return resolved

    def _build_seed_sampler(self) -> RealisticGenomeSeeder | None:
        try:
            return RealisticGenomeSeeder()
        except Exception as exc:
            logger.warning("Realistic genome seeding unavailable: %s", exc)
            return None

    def _seed_with_sampler(
        self, provider: GenomeProvider
    ) -> Tuple[List[DecisionGenome], Dict[str, object] | None]:
        if self._seed_sampler is None:
            return [], None

        seeded: List[DecisionGenome] = []
        try:
            for index in range(self.population_size):
                seed = self._seed_sampler.sample()
                genome = provider.new_genome(
                    id=f"realistic_seed_{index:05d}",
                    parameters=dict(seed.parameters),
                    generation=0,
                    species_type=seed.species,
                )
                resolved = cast(DecisionGenome, provider.from_legacy(genome))
                resolved = cast(DecisionGenome, apply_seed_to_genome(resolved, seed))
                seeded.append(resolved)
        except Exception as exc:
            logger.error("Realistic genome seeding failed: %s", exc)
            return [], None

        summary = summarize_seed_metadata(seeded)
        return seeded, summary

    def evolve_population(self, market_data: Dict, performance_metrics: Dict) -> None:
        """Evolve the population based on market data and performance."""
        try:
            if not self.population:
                self._generate_initial_population()
                return

            # Evaluate fitness for all genomes
            self._evaluate_fitness(market_data, performance_metrics)

            # Select best performers
            elite_count = max(1, int(self.population_size * 0.1))  # Top 10%
            elite_genomes = self.get_best_genomes(elite_count)

            # Generate new population
            new_population: List[DecisionGenome] = cast(
                List[DecisionGenome], elite_genomes.copy()
            )  # Keep elite

            # Fill remaining slots with offspring
            while len(new_population) < self.population_size:
                # Select parents via tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                # Create offspring via crossover and mutation
                offspring = self._crossover(parent1, parent2)
                offspring = self._mutate(offspring)
                new_population.append(cast(DecisionGenome, offspring))

            # Update population
            self.population = new_population
            self.generation += 1

            best = max((g.fitness or 0.0) for g in self.population) if self.population else 0.0
            logger.info(
                f"Evolution complete. Generation {self.generation}, Best fitness: {best:.4f}"
            )

        except Exception as e:
            logger.error(f"Evolution failed: {e}")

    def _evaluate_fitness(self, market_data: Dict, performance_metrics: Dict) -> None:
        """Evaluate fitness for all genomes based on market performance."""
        for genome in self.population:
            try:
                # Calculate fitness based on multiple factors
                profit_score = float(performance_metrics.get("total_return", 0.0)) * 0.4
                risk_score = (1.0 - float(performance_metrics.get("max_drawdown", 1.0))) * 0.3
                consistency_score = float(performance_metrics.get("win_rate", 0.0)) * 0.2
                efficiency_score = float(performance_metrics.get("sharpe_ratio", 0.0)) * 0.1

                genome.fitness = max(
                    0.0, profit_score + risk_score + consistency_score + efficiency_score
                )

                # Update performance metrics with numeric values only
                try:
                    perf_dict = getattr(genome, "performance_metrics", None)
                    if isinstance(perf_dict, dict):
                        for k, v in performance_metrics.items():
                            try:
                                perf_dict[k] = float(v)
                            except Exception:
                                continue
                except Exception:
                    pass

            except Exception as e:
                gid = getattr(genome, "id", "unknown")
                logger.error(f"Fitness evaluation failed for genome {gid}: {e}")
                try:
                    genome.fitness = 0.0
                except Exception:
                    pass

    def _tournament_selection(self, tournament_size: int = 3) -> DecisionGenome:
        """Select genome via tournament selection."""
        import random

        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: (g.fitness or 0.0))

    def _crossover(self, parent1: object, parent2: object) -> object:
        """Create offspring via crossover."""
        import random

        # Create new genome ID
        offspring_id = f"genome_{self.generation}_{random.randint(1000, 9999)}"

        # Blend parameters from both parents
        p1_params = getattr(parent1, "parameters", {}) or {}
        p2_params = getattr(parent2, "parameters", {}) or {}
        if not isinstance(p1_params, dict):
            p1_params = {}
        if not isinstance(p2_params, dict):
            p2_params = {}

        offspring_params: Dict[str, float] = {}
        for key in p1_params:
            try:
                if random.random() < 0.5:
                    offspring_params[key] = float(p1_params[key])
                else:
                    fallback = p1_params[key]
                    val = p2_params.get(key, fallback)
                    offspring_params[key] = float(val)
            except Exception:
                continue

        provider = get_genome_provider()
        try:
            child = provider.new_genome(
                id=offspring_id,
                parameters=offspring_params,
                generation=self.generation,
                species_type=getattr(parent1, "species_type", None),
            )
            # Attempt to attach parent metadata if supported
            try:
                pid1 = getattr(parent1, "id", None)
                pid2 = getattr(parent2, "id", None)
                if hasattr(child, "with_updated"):
                    child = child.with_updated(parent_ids=[pid1, pid2])
            except Exception:
                pass
            return child
        except Exception:
            # Safe fallback to parent1 clone if anything goes wrong
            return parent1

    def _mutate(self, genome: object, mutation_rate: float = 0.1) -> object:
        """Apply mutation to genome."""
        import random

        new_params: Dict[str, float] = {}
        params = getattr(genome, "parameters", {}) or {}
        if isinstance(params, dict):
            for key, value in params.items():
                try:
                    if random.random() < mutation_rate:
                        if isinstance(value, float):
                            # Gaussian mutation for float values
                            mutation_strength = 0.1
                            new_val = max(0.0, float(value) + random.gauss(0, mutation_strength))
                        elif isinstance(value, int):
                            # Integer mutation
                            new_val = float(max(1, int(value) + random.randint(-5, 5)))
                        else:
                            try:
                                v = float(value)
                                mutation_strength = 0.1
                                new_val = max(0.0, v + random.gauss(0, mutation_strength))
                            except Exception:
                                continue
                        new_params[key] = float(new_val)
                except Exception:
                    continue

        if not new_params:
            return genome

        # Ensure mutation tags reflect the manager's current generation
        try:
            if hasattr(genome, "with_updated"):
                working = genome.with_updated(generation=self.generation)
            else:
                working = genome
        except Exception:
            working = genome

        # Record mutation using canonical builder (appends canonical tags)
        provider = get_genome_provider()
        try:
            return provider.mutate(working, "gaussian", new_params)
        except Exception:
            return genome


def create_population_manager(
    population_size: int = 100, cache_ttl: int = 300
) -> PopulationManager:
    """Factory helper mirroring legacy construction patterns."""

    return PopulationManager(population_size=population_size, cache_ttl=cache_ttl)
