"""
EvolutionEngine: Evolutionary algorithm for trading strategy optimization.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import random
import uuid

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """Configuration for evolution engine"""
    population_size: int = 500
    elite_ratio: float = 0.1
    crossover_ratio: float = 0.6
    mutation_ratio: float = 0.3
    mutation_rate: float = 0.1
    max_stagnation: int = 20
    complexity_penalty: float = 0.01
    min_fitness_improvement: float = 0.001

@dataclass
class GenerationStats:
    """Statistics for a generation"""
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    stagnation_count: int
    elite_count: int
    new_genomes: int
    complexity_stats: Dict[str, float]

class EvolutionEngine:
    """
    Evolutionary algorithm engine for optimizing trading strategies.
    
    Manages a population of DecisionGenomes and evolves them through
    selection, crossover, and mutation to find optimal trading strategies.
    """
    
    def __init__(self, config: EvolutionConfig, fitness_evaluator):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        
        # Population management
        self.population: List = []
        self.best_genome = None
        self.best_fitness = 0.0
        
        # Evolution tracking
        self.generation = 0
        self.stagnation_count = 0
        self.generation_history: List[GenerationStats] = []
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = timedelta(0)
        
        # Diversity tracking
        self.diversity_history: List[float] = []
        
        logger.info(f"Initialized EvolutionEngine with population size {config.population_size}")
    
    def initialize_population(self, seed: Optional[int] = None) -> bool:
        """
        Initialize the population with random genomes.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            True if initialization successful
        """
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            
            logger.info(f"Initializing population of {self.config.population_size} genomes")
            
            # Create initial population
            from emp.agent.genome import DecisionGenome
            
            self.population = []
            for i in range(self.config.population_size):
                genome = DecisionGenome(max_depth=10, max_nodes=100)
                genome.generation = 0
                self.population.append(genome)
            
            self.generation = 0
            self.stagnation_count = 0
            
            logger.info(f"Population initialized with {len(self.population)} genomes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize population: {e}")
            return False
    
    def evolve_generation(self) -> GenerationStats:
        """
        Evolve the population to the next generation.
        
        Returns:
            GenerationStats for the new generation
        """
        logger.info(f"Evolving generation {self.generation}")
        
        # Evaluate current population
        self._evaluate_population()
        
        # Update best genome
        self._update_best_genome()
        
        # Check for stagnation
        self._check_stagnation()
        
        # Create next generation
        self._create_next_generation()
        
        # Calculate generation statistics
        stats = self._calculate_generation_stats()
        self.generation_history.append(stats)
        
        self.generation += 1
        
        logger.info(f"Generation {self.generation - 1} evolved. Best fitness: {self.best_fitness:.4f}")
        
        return stats
    
    def _evaluate_population(self):
        """Evaluate fitness for all genomes in the population."""
        logger.info(f"Evaluating {len(self.population)} genomes")
        
        evaluation_start = datetime.now()
        
        for i, genome in enumerate(self.population):
            try:
                # Evaluate genome
                fitness_score = self.fitness_evaluator.evaluate_genome(genome)
                genome.fitness_score = fitness_score.total_fitness
                
                # Update evaluation count
                self.evaluation_count += 1
                
                if i % 50 == 0:
                    logger.info(f"Evaluated {i}/{len(self.population)} genomes")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate genome {genome.genome_id}: {e}")
                genome.fitness_score = 0.0
        
        evaluation_time = datetime.now() - evaluation_start
        self.total_evaluation_time += evaluation_time
        
        logger.info(f"Population evaluation completed in {evaluation_time}")
    
    def _update_best_genome(self):
        """Update the best genome in the population."""
        if not self.population:
            return
        
        # Find genome with highest fitness
        best_genome = max(self.population, key=lambda g: g.fitness_score)
        
        if best_genome.fitness_score > self.best_fitness:
            self.best_genome = best_genome
            self.best_fitness = best_genome.fitness_score
            self.stagnation_count = 0
            logger.info(f"New best genome found: {best_genome.fitness_score:.4f}")
        else:
            self.stagnation_count += 1
    
    def _check_stagnation(self):
        """Check if evolution has stagnated."""
        if self.stagnation_count >= self.config.max_stagnation:
            logger.warning(f"Evolution stagnated for {self.stagnation_count} generations")
            
            # Increase mutation rate temporarily
            self.config.mutation_rate = min(self.config.mutation_rate * 1.5, 0.5)
            logger.info(f"Increased mutation rate to {self.config.mutation_rate:.3f}")
    
    def _create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        # Calculate counts for different reproduction methods
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        crossover_count = int(self.config.population_size * self.config.crossover_ratio)
        mutation_count = self.config.population_size - elite_count - crossover_count
        
        # Ensure we have enough genomes
        if elite_count + crossover_count + mutation_count > self.config.population_size:
            mutation_count = self.config.population_size - elite_count - crossover_count
        
        logger.info(f"Creating next generation: {elite_count} elite, {crossover_count} crossover, {mutation_count} mutation")
        
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        new_population = []
        
        # Elitism: Keep best genomes
        elite_genomes = self.population[:elite_count]
        new_population.extend(elite_genomes)
        
        # Crossover: Create offspring from parent pairs
        crossover_offspring = self._create_crossover_offspring(crossover_count)
        new_population.extend(crossover_offspring)
        
        # Mutation: Create mutated versions of existing genomes
        mutation_offspring = self._create_mutation_offspring(mutation_count)
        new_population.extend(mutation_offspring)
        
        # Update population
        self.population = new_population[:self.config.population_size]
        
        # Reset mutation rate if it was increased
        if self.stagnation_count < self.config.max_stagnation:
            self.config.mutation_rate = 0.1  # Reset to default
    
    def _create_crossover_offspring(self, count: int) -> List:
        """Create offspring through crossover."""
        offspring = []
        
        # Create tournament selection pool
        tournament_pool = self.population.copy()
        
        for i in range(count):
            try:
                # Select two parents through tournament selection
                parent1 = self._tournament_selection(tournament_pool)
                parent2 = self._tournament_selection(tournament_pool)
                
                if parent1 and parent2:
                    # Perform crossover
                    offspring1, offspring2 = parent1.crossover(parent2)
                    
                    # Add to offspring list
                    if len(offspring) < count:
                        offspring.append(offspring1)
                    if len(offspring) < count:
                        offspring.append(offspring2)
                
            except Exception as e:
                logger.error(f"Crossover failed: {e}")
                # Create a random genome as fallback
                from emp.agent.genome import DecisionGenome
                fallback_genome = DecisionGenome(max_depth=10, max_nodes=100)
                fallback_genome.generation = self.generation + 1
                offspring.append(fallback_genome)
        
        return offspring
    
    def _create_mutation_offspring(self, count: int) -> List:
        """Create offspring through mutation."""
        offspring = []
        
        # Create tournament selection pool
        tournament_pool = self.population.copy()
        
        for i in range(count):
            try:
                # Select parent through tournament selection
                parent = self._tournament_selection(tournament_pool)
                
                if parent:
                    # Perform mutation
                    mutated_genome = parent.mutate(self.config.mutation_rate)
                    offspring.append(mutated_genome)
                
            except Exception as e:
                logger.error(f"Mutation failed: {e}")
                # Create a random genome as fallback
                from emp.agent.genome import DecisionGenome
                fallback_genome = DecisionGenome(max_depth=10, max_nodes=100)
                fallback_genome.generation = self.generation + 1
                offspring.append(fallback_genome)
        
        return offspring
    
    def _tournament_selection(self, pool: List, tournament_size: int = 3) -> Optional:
        """Select a genome using tournament selection."""
        if not pool:
            return None
        
        # Select random participants
        participants = random.sample(pool, min(tournament_size, len(pool)))
        
        # Return the best participant
        return max(participants, key=lambda g: g.fitness_score)
    
    def _apply_complexity_constraints(self, population: List) -> List:
        """Apply complexity constraints to the population."""
        # Filter out overly complex genomes
        max_complexity = 200  # Maximum total nodes
        max_depth = 15       # Maximum tree depth
        
        filtered_population = []
        
        for genome in population:
            complexity = genome.get_complexity()
            nodes = complexity.get('nodes', 0)
            depth = complexity.get('depth', 0)
            
            if nodes <= max_complexity and depth <= max_depth:
                filtered_population.append(genome)
            else:
                # Apply complexity penalty
                genome.fitness_score *= (1.0 - self.config.complexity_penalty)
                filtered_population.append(genome)
        
        return filtered_population
    
    def _calculate_generation_stats(self) -> GenerationStats:
        """Calculate statistics for the current generation."""
        if not self.population:
            return GenerationStats(
                generation=self.generation,
                population_size=0,
                best_fitness=0.0,
                average_fitness=0.0,
                worst_fitness=0.0,
                diversity_score=0.0,
                stagnation_count=self.stagnation_count,
                elite_count=0,
                new_genomes=0,
                complexity_stats={}
            )
        
        # Basic fitness statistics
        fitness_scores = [g.fitness_score for g in self.population]
        best_fitness = max(fitness_scores)
        average_fitness = np.mean(fitness_scores)
        worst_fitness = min(fitness_scores)
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score()
        self.diversity_history.append(diversity_score)
        
        # Complexity statistics
        complexity_stats = self._calculate_complexity_stats()
        
        # Count elite and new genomes
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        new_genomes = len([g for g in self.population if g.generation == self.generation])
        
        return GenerationStats(
            generation=self.generation,
            population_size=len(self.population),
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            diversity_score=diversity_score,
            stagnation_count=self.stagnation_count,
            elite_count=elite_count,
            new_genomes=new_genomes,
            complexity_stats=complexity_stats
        )
    
    def _calculate_diversity_score(self) -> float:
        """Calculate population diversity score."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate fitness diversity
        fitness_scores = [g.fitness_score for g in self.population]
        fitness_std = np.std(fitness_scores)
        fitness_mean = np.mean(fitness_scores)
        
        if fitness_mean == 0:
            return 0.0
        
        # Coefficient of variation
        cv = fitness_std / fitness_mean
        
        # Normalize to 0-1 range
        diversity_score = min(cv, 1.0)
        
        return diversity_score
    
    def _calculate_complexity_stats(self) -> Dict[str, float]:
        """Calculate complexity statistics for the population."""
        if not self.population:
            return {}
        
        all_nodes = []
        all_depths = []
        all_leaves = []
        
        for genome in self.population:
            complexity = genome.get_complexity()
            all_nodes.append(complexity.get('nodes', 0))
            all_depths.append(complexity.get('depth', 0))
            all_leaves.append(complexity.get('leaves', 0))
        
        return {
            'avg_nodes': np.mean(all_nodes),
            'avg_depth': np.mean(all_depths),
            'avg_leaves': np.mean(all_leaves),
            'max_nodes': np.max(all_nodes),
            'max_depth': np.max(all_depths)
        }
    
    def get_population_summary(self) -> Dict:
        """Get summary of the current population."""
        if not self.population:
            return {}
        
        fitness_scores = [g.fitness_score for g in self.population]
        complexity_stats = self._calculate_complexity_stats()
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'best_fitness': max(fitness_scores),
            'average_fitness': np.mean(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'diversity_score': self._calculate_diversity_score(),
            'complexity_stats': complexity_stats,
            'evaluation_count': self.evaluation_count,
            'total_evaluation_time': str(self.total_evaluation_time)
        }
    
    def get_best_genomes(self, count: int = 10) -> List:
        """Get the best genomes from the population."""
        if not self.population:
            return []
        
        # Sort by fitness and return top genomes
        sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_population[:count]
    
    def save_population(self, filepath: str):
        """Save the current population to a file."""
        try:
            import json
            
            population_data = []
            for genome in self.population:
                genome_dict = genome.to_dict()
                population_data.append(genome_dict)
            
            data = {
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'stagnation_count': self.stagnation_count,
                'population': population_data
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Population saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save population: {e}")
    
    def load_population(self, filepath: str):
        """Load a population from a file."""
        try:
            import json
            from emp.agent.genome import DecisionGenome
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load population
            self.population = []
            for genome_data in data.get('population', []):
                genome = DecisionGenome(max_depth=10, max_nodes=100)
                genome.from_dict(genome_data)
                self.population.append(genome)
            
            # Load metadata
            self.generation = data.get('generation', 0)
            self.best_fitness = data.get('best_fitness', 0.0)
            self.stagnation_count = data.get('stagnation_count', 0)
            
            # Update best genome
            if self.population:
                self.best_genome = max(self.population, key=lambda g: g.fitness_score)
            
            logger.info(f"Population loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load population: {e}")
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of the evolution process."""
        if not self.generation_history:
            return {}
        
        # Calculate evolution metrics
        generations = len(self.generation_history)
        best_fitness_history = [stats.best_fitness for stats in self.generation_history]
        avg_fitness_history = [stats.average_fitness for stats in self.generation_history]
        
        # Calculate improvement
        if len(best_fitness_history) > 1:
            total_improvement = best_fitness_history[-1] - best_fitness_history[0]
            avg_improvement_per_gen = total_improvement / (generations - 1)
        else:
            total_improvement = 0.0
            avg_improvement_per_gen = 0.0
        
        return {
            'total_generations': generations,
            'current_generation': self.generation,
            'best_fitness_ever': max(best_fitness_history) if best_fitness_history else 0.0,
            'current_best_fitness': best_fitness_history[-1] if best_fitness_history else 0.0,
            'total_improvement': total_improvement,
            'avg_improvement_per_gen': avg_improvement_per_gen,
            'stagnation_count': self.stagnation_count,
            'evaluation_count': self.evaluation_count,
            'total_evaluation_time': str(self.total_evaluation_time),
            'diversity_trend': self.diversity_history[-10:] if self.diversity_history else []
        } 