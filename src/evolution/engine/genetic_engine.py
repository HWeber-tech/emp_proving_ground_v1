"""
EMP Genetic Engine v1.1

Lean orchestrator for the evolutionary process.
Uses dependency injection to coordinate modular components.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.interfaces import (
    IPopulationManager,
    ISelectionStrategy,
    ICrossoverStrategy,
    IMutationStrategy,
    IFitnessEvaluator,
    IGenomeFactory,
    IEvolutionLogger
)
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the genetic engine."""
    population_size: int = 100
    elite_count: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    max_generations: int = 100
    target_fitness: Optional[float] = None
    early_stopping_generations: int = 10


@dataclass
class EvolutionStats:
    """Statistics for a generation."""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    fitness_std: float
    diversity_score: float
    elapsed_time: float


class GeneticEngine:
    """
    Lean orchestrator for the evolutionary process.
    
    This class coordinates the evolutionary process using injected components.
    It delegates all specific operations to specialized modules and focuses
    on orchestration and state management.
    """
    
    def __init__(
        self,
        population_manager: IPopulationManager,
        selection_strategy: ISelectionStrategy,
        crossover_strategy: ICrossoverStrategy,
        mutation_strategy: IMutationStrategy,
        fitness_evaluator: IFitnessEvaluator,
        genome_factory: IGenomeFactory,
        evolution_logger: Optional[IEvolutionLogger] = None,
        config: Optional[EvolutionConfig] = None
    ):
        """
        Initialize the genetic engine with injected dependencies.
        
        Args:
            population_manager: Manages the population of genomes
            selection_strategy: Strategy for selecting parents
            crossover_strategy: Strategy for creating offspring
            mutation_strategy: Strategy for mutating genomes
            fitness_evaluator: Evaluates genome fitness
            genome_factory: Creates new genomes
            evolution_logger: Optional logger for evolution progress
            config: Evolution configuration
        """
        self.population_manager = population_manager
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.fitness_evaluator = fitness_evaluator
        self.genome_factory = genome_factory
        self.evolution_logger = evolution_logger
        self.config = config or EvolutionConfig()
        
        self._best_genome: Optional[DecisionGenome] = None
        self._best_fitness: float = float('-inf')
        self._generation_history: List[EvolutionStats] = []
        
        logger.info("GeneticEngine initialized with dependency injection")
    
    def initialize_population(self) -> None:
        """Initialize the population using the genome factory."""
        self.population_manager.initialize_population(self.genome_factory.create_genome)
        self._evaluate_population()
        logger.info("Population initialized")
    
    def evolve_generation(self) -> EvolutionStats:
        """
        Evolve one generation using the evolutionary process.
        
        Returns:
            Statistics for this generation
        """
        start_time = datetime.now()
        
        # Get current population and fitness scores
        population = self.population_manager.get_population()
        fitness_scores = [genome.fitness_score for genome in population]
        
        # Create new population
        new_population = []
        
        # Preserve elite genomes
        elite_genomes = self.population_manager.get_best_genomes(self.config.elite_count)
        new_population.extend(elite_genomes)
        
        # Generate offspring
        offspring_needed = self.config.population_size - len(new_population)
        
        for _ in range(offspring_needed):
            # Select parents
            parent1 = self.selection_strategy.select(population, fitness_scores)
            parent2 = self.selection_strategy.select(population, fitness_scores)
            
            # Create offspring
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover_strategy.crossover(parent1, parent2)
                child = child1  # Use first child
            else:
                child = parent1  # Clone parent
            
            # Mutate child
            if random.random() < self.config.mutation_rate:
                child = self.mutation_strategy.mutate(child, self.config.mutation_rate)
                child.mutation_count += 1
            
            # Update generation and parent info
            child.generation = self.population_manager.generation + 1
            child.parent_ids = [parent1.genome_id, parent2.genome_id]
            
            new_population.append(child)
        
        # Update population
        self.population_manager.update_population(new_population)
        
        # Evaluate new population
        self._evaluate_population()
        
        # Update best genome
        self._update_best_genome()
        
        # Advance generation
        self.population_manager.advance_generation()
        
        # Create stats
        stats = self._create_generation_stats(start_time)
        self._generation_history.append(stats)
        
        # Log if logger provided
        if self.evolution_logger:
            self.evolution_logger.log_generation(
                stats.generation,
                self.population_manager.get_population_statistics()
            )
        
        logger.debug(f"Evolved generation {stats.generation}: best_fitness={stats.best_fitness:.4f}")
        
        return stats
    
    def evolve(self, max_generations: Optional[int] = None) -> List[EvolutionStats]:
        """
        Evolve multiple generations.
        
        Args:
            max_generations: Maximum generations to evolve (uses config if None)
            
        Returns:
            List of generation statistics
        """
        max_gens = max_generations or self.config.max_generations
        
        logger.info(f"Starting evolution for {max_gens} generations")
        
        for generation in range(max_gens):
            stats = self.evolve_generation()
            
            # Check for early stopping
            if self._should_stop_early(stats):
                logger.info(f"Early stopping at generation {generation + 1}")
                break
            
            # Check target fitness
            if (self.config.target_fitness and 
                stats.best_fitness >= self.config.target_fitness):
                logger.info(f"Target fitness reached at generation {generation + 1}")
                break
        
        logger.info(f"Evolution completed after {len(self._generation_history)} generations")
        
        return self._generation_history
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all genomes in the population."""
        population = self.population_manager.get_population()
        
        for genome in population:
            if genome.fitness_score == 0.0:  # Only evaluate if not already evaluated
                fitness = self.fitness_evaluator.evaluate(genome)
                genome.fitness_score = fitness
    
    def _update_best_genome(self) -> None:
        """Update the best genome found so far."""
        best_genomes = self.population_manager.get_best_genomes(1)
        if best_genomes:
            best = best_genomes[0]
            if best.fitness_score > self._best_fitness:
                self._best_genome = best
                self._best_fitness = best.fitness_score
                
                if self.evolution_logger:
                    self.evolution_logger.log_best_genome(best, best.fitness_score)
    
    def _create_generation_stats(self, start_time: datetime) -> EvolutionStats:
        """Create statistics for the current generation."""
        stats = self.population_manager.get_population_statistics()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        return EvolutionStats(
            generation=stats['generation'],
            best_fitness=stats['best_fitness'],
            average_fitness=stats['average_fitness'],
            worst_fitness=stats['worst_fitness'],
            fitness_std=stats['fitness_std'],
            diversity_score=stats['diversity_score'],
            elapsed_time=elapsed_time
        )
    
    def _should_stop_early(self, stats: EvolutionStats) -> bool:
        """Check if evolution should stop early."""
        if len(self._generation_history) < self.config.early_stopping_generations:
            return False
        
        # Check if improvement has plateaued
        recent_best = [g.best_fitness for g in self._generation_history[-self.config.early_stopping_generations:]]
        if len(recent_best) >= 2:
            improvement = recent_best[-1] - recent_best[0]
            return improvement < 0.001  # Very small improvement threshold
        
        return False
    
    def get_best_genome(self) -> Optional[DecisionGenome]:
        """Get the best genome found so far."""
        return self._best_genome
    
    def get_best_fitness(self) -> float:
        """Get the best fitness score found so far."""
        return self._best_fitness
    
    def get_generation_history(self) -> List[EvolutionStats]:
        """Get the complete evolution history."""
        return self._generation_history.copy()
    
    def reset(self) -> None:
        """Reset the genetic engine to initial state."""
        self.population_manager.reset()
        self._best_genome = None
        self._best_fitness = float('-inf')
        self._generation_history.clear()
        logger.info("GeneticEngine reset to initial state")
    
    def __repr__(self) -> str:
        """String representation of the genetic engine."""
        return (f"GeneticEngine(generation={self.population_manager.generation}, "
                f"best_fitness={self._best_fitness:.4f}, "
                f"population_size={self.population_manager.population_size})")
