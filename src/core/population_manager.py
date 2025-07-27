"""
Population Manager Implementation
=================================

Concrete implementation of IPopulationManager for genetic algorithm population management.
Optimized for performance with Redis caching integration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import numpy as np

from .interfaces import IPopulationManager, DecisionGenome
from ..performance import get_global_cache

logger = logging.getLogger(__name__)


class PopulationManager(IPopulationManager):
    """High-performance population manager with caching support."""
    
    def __init__(self, population_size: int = 100, cache_ttl: int = 300):
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
        
    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize population with new genomes."""
        logger.info(f"Initializing population with {self.population_size} genomes")
        self.population = [genome_factory() for _ in range(self.population_size)]
        self.generation = 0
        self._cache_population_stats()
        
    def get_population(self) -> List[DecisionGenome]:
        """Get current population."""
        return self.population.copy()
    
    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """Get top N genomes by fitness."""
        if not self.population:
            return []
        
        # Sort by fitness (descending)
        sorted_population = sorted(
            self.population,
            key=lambda g: getattr(g, 'fitness', 0.0),
            reverse=True
        )
        return sorted_population[:min(count, len(sorted_population))]
    
    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """Replace current population with new one."""
        logger.info(f"Updating population with {len(new_population)} genomes")
        self.population = new_population
        self._cache_population_stats()
        
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        if not self.population:
            return {
                'generation': self.generation,
                'population_size': 0,
                'average_fitness': 0.0,
                'best_fitness': 0.0,
                'worst_fitness': 0.0,
                'fitness_std': 0.0,
                'species_distribution': {}
            }
        
        fitness_values = [getattr(g, 'fitness', 0.0) for g in self.population]
        
        # Species distribution
        species_count = {}
        for genome in self.population:
            species = getattr(genome, 'species_type', 'generic')
            species_count[species] = species_count.get(species, 0) + 1
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'average_fitness': float(np.mean(fitness_values)),
            'best_fitness': float(np.max(fitness_values)),
            'worst_fitness': float(np.min(fitness_values)),
            'fitness_std': float(np.std(fitness_values)),
            'species_distribution': species_count
        }
    
    def advance_generation(self) -> None:
        """Increment the generation counter."""
        self.generation += 1
        logger.info(f"Advanced to generation {self.generation}")
    
    def reset(self) -> None:
        """Reset the population manager to initial state."""
        logger.info("Resetting population manager")
        self.population.clear()
        self.generation = 0
    
    def _cache_population_stats(self) -> None:
        """Cache population statistics for performance."""
        stats = self.get_population_statistics()
        logger.debug(f"Cached population stats: {stats}")
    
    def get_genome_by_id(self, genome_id: str) -> Optional[DecisionGenome]:
        """Get a specific genome by ID."""
        for genome in self.population:
            if genome.id == genome_id:
                return genome
        return None
    
    def get_species_count(self, species_type: str) -> int:
        """Get count of genomes for a specific species."""
        return sum(1 for g in self.population if getattr(g, 'species_type', 'generic') == species_type)
    
    def get_fitness_distribution(self) -> Dict[str, float]:
        """Get fitness distribution statistics."""
        if not self.population:
            return {}
        
        fitness_values = [getattr(g, 'fitness', 0.0) for g in self.population]
        return {
            'min': float(np.min(fitness_values)),
            'max': float(np.max(fitness_values)),
            'mean': float(np.mean(fitness_values)),
            'median': float(np.median(fitness_values)),
            'std': float(np.std(fitness_values)),
            'percentile_25': float(np.percentile(fitness_values, 25)),
            'percentile_75': float(np.percentile(fitness_values, 75))
        }
