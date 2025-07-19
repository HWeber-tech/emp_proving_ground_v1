"""
EMP Population Manager v1.1

Single-responsibility population management for the evolutionary process.
Encapsulates all population-related operations and state management.
"""

import logging
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from statistics import mean, stdev

from src.genome.models.genome import DecisionGenome
from src.core.interfaces import IPopulationManager

logger = logging.getLogger(__name__)


@dataclass
class PopulationStatistics:
    """Statistics about the current population."""
    size: int
    generation: int
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    fitness_std: float
    fitness_range: float
    diversity_score: float


class PopulationManager(IPopulationManager):
    """
    Manages the population of genomes in the evolutionary process.
    
    This class encapsulates all population-related operations, including
    initialization, state management, and lifecycle tracking. It serves
    as the single source of truth for population state.
    """
    
    def __init__(self, population_size: int):
        """
        Initialize the population manager.
        
        Args:
            population_size: Fixed size for the population
        """
        self._population_size = population_size
        self._population: List[DecisionGenome] = []
        self._generation = 0
        self._fitness_cache: Dict[str, float] = {}
        
        logger.info(f"PopulationManager initialized with size {population_size}")
    
    def initialize_population(self, genome_factory: Callable[[], DecisionGenome]) -> None:
        """
        Initialize the population with new genomes.
        
        Args:
            genome_factory: Callable that creates new genomes
        """
        self._population = [genome_factory() for _ in range(self._population_size)]
        self._generation = 0
        self._fitness_cache.clear()
        
        logger.info(f"Initialized population with {len(self._population)} genomes")
    
    def get_population(self) -> List[DecisionGenome]:
        """Get the current population."""
        return self._population.copy()
    
    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """
        Get the top N genomes by fitness.
        
        Args:
            count: Number of top genomes to return
            
        Returns:
            List of best genomes, sorted by fitness descending
        """
        if not self._population:
            return []
        
        # Sort by fitness (assuming fitness is stored in genome)
        sorted_population = sorted(
            self._population,
            key=lambda g: getattr(g, 'fitness', 0.0),
            reverse=True
        )
        
        return sorted_population[:min(count, len(sorted_population))]
    
    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """
        Replace the current population with a new one.
        
        Args:
            new_population: New population to set
            
        Raises:
            ValueError: If new population size doesn't match configured size
        """
        if len(new_population) != self._population_size:
            raise ValueError(
                f"New population size {len(new_population)} "
                f"must match configured size {self._population_size}"
            )
        
        self._population = new_population.copy()
        self._fitness_cache.clear()
        
        logger.debug(f"Updated population with {len(new_population)} genomes")
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        if not self._population:
            return {
                'size': 0,
                'generation': self._generation,
                'best_fitness': 0.0,
                'worst_fitness': 0.0,
                'average_fitness': 0.0,
                'fitness_std': 0.0,
                'fitness_range': 0.0,
                'diversity_score': 0.0
            }
        
        # Extract fitness scores
        fitness_scores = [
            getattr(genome, 'fitness', 0.0) 
            for genome in self._population
        ]
        
        if not fitness_scores:
            fitness_scores = [0.0] * len(self._population)
        
        best_fitness = max(fitness_scores)
        worst_fitness = min(fitness_scores)
        avg_fitness = mean(fitness_scores)
        
        # Calculate standard deviation
        if len(fitness_scores) > 1:
            fitness_std = stdev(fitness_scores)
        else:
            fitness_std = 0.0
        
        fitness_range = best_fitness - worst_fitness
        
        # Calculate diversity score (simplified)
        diversity_score = self._calculate_diversity_score()
        
        return {
            'size': len(self._population),
            'generation': self._generation,
            'best_fitness': best_fitness,
            'worst_fitness': worst_fitness,
            'average_fitness': avg_fitness,
            'fitness_std': fitness_std,
            'fitness_range': fitness_range,
            'diversity_score': diversity_score
        }
    
    def _calculate_diversity_score(self) -> float:
        """Calculate a simple diversity score based on genome similarity."""
        if len(self._population) < 2:
            return 0.0
        
        try:
            # Calculate diversity based on key parameter differences
            strategy_params = []
            risk_params = []
            
            for genome in self._population:
                # Extract key strategy parameters
                strategy_params.append([
                    genome.strategy.entry_threshold,
                    genome.strategy.exit_threshold,
                    genome.strategy.momentum_weight,
                    genome.strategy.trend_weight,
                    genome.strategy.lookback_period
                ])
                
                # Extract key risk parameters
                risk_params.append([
                    genome.risk.risk_tolerance,
                    genome.risk.stop_loss_threshold,
                    genome.risk.take_profit_threshold,
                    genome.risk.max_drawdown_limit
                ])
            
            # Calculate average parameter variance across all dimensions
            all_variances = []
            
            # Strategy parameter variances
            if strategy_params and strategy_params[0]:
                for i in range(len(strategy_params[0])):
                    values = [params[i] for params in strategy_params]
                    if values and len(values) > 1:
                        all_variances.append(stdev(values))
            
            # Risk parameter variances
            if risk_params and risk_params[0]:
                for i in range(len(risk_params[0])):
                    values = [params[i] for params in risk_params]
                    if values and len(values) > 1:
                        all_variances.append(stdev(values))
            
            if all_variances:
                avg_variance = mean(all_variances)
                # Normalize to 0-1 range based on typical parameter ranges
                max_possible_variance = 0.5  # Adjust based on expected parameter ranges
                return min(avg_variance / max_possible_variance, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating diversity score: {e}")
        
        return 0.0
    
    def cache_fitness(self, genome_id: str, fitness: float) -> None:
        """Cache fitness score for a genome."""
        self._fitness_cache[genome_id] = fitness
    
    def get_cached_fitness(self, genome_id: str) -> Optional[float]:
        """Get cached fitness score for a genome."""
        return self._fitness_cache.get(genome_id)
    
    def clear_fitness_cache(self) -> None:
        """Clear the fitness cache."""
        self._fitness_cache.clear()
    
    @property
    def population_size(self) -> int:
        """Return the configured population size."""
        return self._population_size
    
    @property
    def generation(self) -> int:
        """Return the current generation number."""
        return self._generation
    
    def advance_generation(self) -> None:
        """Increment the generation counter."""
        self._generation += 1
        logger.debug(f"Advanced to generation {self._generation}")
    
    def reset(self) -> None:
        """Reset the population manager to initial state."""
        self._population.clear()
        self._generation = 0
        self._fitness_cache.clear()
        logger.info("PopulationManager reset to initial state")
    
    def __len__(self) -> int:
        """Return the current population size."""
        return len(self._population)
    
    def __repr__(self) -> str:
        """String representation of the population manager."""
        return (f"PopulationManager(size={self._population_size}, "
                f"generation={self._generation}, "
                f"current_population={len(self._population)})")
