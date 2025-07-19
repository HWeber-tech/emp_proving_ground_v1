"""
EMP Population Manager v1.1

Population management for the adaptive core.
Handles population lifecycle, diversity, and management.
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import logging

from ...genome.models.genome import StrategyGenome
from ...core.events import FitnessReport, EvolutionEvent
from ...core.event_bus import publish_event, EventType

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages population lifecycle and diversity."""
    
    def __init__(self, population_size: int = 100, elite_size: int = 10):
        self.population_size = population_size
        self.elite_size = elite_size
        self.population: List[StrategyGenome] = []
        self.fitness_cache: Dict[str, float] = {}
        self.generation = 0
        self.population_history: List[Dict[str, Any]] = []
        
        logger.info(f"Population Manager initialized with size {population_size}")
        
    def initialize_population(self, genome_factory: Callable[[], StrategyGenome]):
        """Initialize population with random genomes."""
        self.population = []
        for i in range(self.population_size):
            genome = genome_factory()
            genome.id = f"genome_gen{self.generation}_id{i}"
            self.population.append(genome)
            
        logger.info(f"Initialized population with {len(self.population)} genomes")
        
    def update_fitness_cache(self, fitness_reports: List[FitnessReport]):
        """Update fitness cache with new fitness reports."""
        for report in fitness_reports:
            self.fitness_cache[report.genome_id] = report.fitness_score
            
        logger.debug(f"Updated fitness cache with {len(fitness_reports)} reports")
        
    def get_fitness(self, genome_id: str) -> float:
        """Get fitness score for a genome."""
        return self.fitness_cache.get(genome_id, 0.0)
        
    def get_best_genomes(self, count: Optional[int] = None) -> List[StrategyGenome]:
        """Get the best genomes based on fitness."""
        if count is None:
            count = self.elite_size
            
        # Sort population by fitness
        sorted_population = sorted(
            self.population, 
            key=lambda g: self.get_fitness(g.id), 
            reverse=True
        )
        
        return sorted_population[:count]
        
    def get_worst_genomes(self, count: int) -> List[StrategyGenome]:
        """Get the worst genomes based on fitness."""
        # Sort population by fitness
        sorted_population = sorted(
            self.population, 
            key=lambda g: self.get_fitness(g.id)
        )
        
        return sorted_population[:count]
        
    def replace_genomes(self, new_genomes: List[StrategyGenome], 
                       replace_indices: List[int]):
        """Replace genomes at specified indices."""
        for i, genome in zip(replace_indices, new_genomes):
            if 0 <= i < len(self.population):
                self.population[i] = genome
                
        logger.debug(f"Replaced {len(new_genomes)} genomes in population")
        
    def add_genomes(self, new_genomes: List[StrategyGenome]):
        """Add new genomes to population."""
        self.population.extend(new_genomes)
        logger.debug(f"Added {len(new_genomes)} genomes to population")
        
    def remove_genomes(self, genome_ids: List[str]):
        """Remove genomes by ID."""
        self.population = [g for g in self.population if g.id not in genome_ids]
        logger.debug(f"Removed {len(genome_ids)} genomes from population")
        
    def calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
            
        # Calculate average pairwise distance
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_genome_distance(
                    self.population[i], self.population[j]
                )
                distances.append(distance)
                
        return np.mean(distances) if distances else 0.0
        
    def _calculate_genome_distance(self, genome1: StrategyGenome, 
                                 genome2: StrategyGenome) -> float:
        """Calculate distance between two genomes."""
        # Euclidean distance between parameter vectors
        params1 = np.array(genome1.parameters)
        params2 = np.array(genome2.parameters)
        
        return np.linalg.norm(params1 - params2)
        
    def calculate_average_fitness(self) -> float:
        """Calculate average fitness of population."""
        if not self.population:
            return 0.0
            
        fitness_values = [self.get_fitness(g.id) for g in self.population]
        return np.mean(fitness_values)
        
    def calculate_fitness_std(self) -> float:
        """Calculate standard deviation of fitness."""
        if not self.population:
            return 0.0
            
        fitness_values = [self.get_fitness(g.id) for g in self.population]
        return np.std(fitness_values)
        
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        if not self.population:
            return {}
            
        fitness_values = [self.get_fitness(g.id) for g in self.population]
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'average_fitness': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'best_fitness': max(fitness_values),
            'worst_fitness': min(fitness_values),
            'diversity': self.calculate_diversity(),
            'elite_size': self.elite_size,
            'fitness_cache_size': len(self.fitness_cache)
        }
        
    def advance_generation(self):
        """Advance to next generation."""
        self.generation += 1
        
        # Log population state
        stats = self.get_population_statistics()
        self.population_history.append({
            'generation': self.generation,
            'timestamp': datetime.now(),
            'statistics': stats
        })
        
        # Publish evolution event
        event = EvolutionEvent(
            timestamp=datetime.now(),
            event_type="generation_complete",
            genome_id="",  # Will be set by caller
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=stats['best_fitness'],
            average_fitness=stats['average_fitness'],
            metadata=stats
        )
        
        # Note: publish_event is async, but we're calling it synchronously here
        # In a real implementation, this would be handled properly
        logger.info(f"Advanced to generation {self.generation}")
        
    def get_population_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get population history."""
        if limit:
            return self.population_history[-limit:]
        return self.population_history.copy()
        
    def reset_population(self):
        """Reset population and clear history."""
        self.population = []
        self.fitness_cache = {}
        self.generation = 0
        self.population_history = []
        logger.info("Population reset")
        
    def export_population(self) -> Dict[str, Any]:
        """Export current population state."""
        return {
            'population': [g.to_dict() for g in self.population],
            'fitness_cache': self.fitness_cache,
            'generation': self.generation,
            'statistics': self.get_population_statistics()
        }
        
    def import_population(self, data: Dict[str, Any]):
        """Import population state."""
        self.population = [StrategyGenome.from_dict(g) for g in data.get('population', [])]
        self.fitness_cache = data.get('fitness_cache', {})
        self.generation = data.get('generation', 0)
        logger.info(f"Imported population with {len(self.population)} genomes") 