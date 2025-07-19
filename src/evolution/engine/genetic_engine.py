"""
EMP Genetic Engine v1.1

Core genetic algorithm engine for evolving trading strategies
and optimizing system parameters across all layers.
"""

import asyncio
import logging
import random
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from src.genome.models.genome import DecisionGenome
from src.core.interfaces import SensorySignal, AnalysisResult
from src.core.exceptions import EvolutionException
from src.core.event_bus import event_bus

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    max_generations: int = 100
    fitness_threshold: float = 0.8
    diversity_threshold: float = 0.3
    convergence_patience: int = 10


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    timestamp: datetime
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    mutation_count: int
    crossover_count: int
    elite_count: int


class GeneticEngine:
    """Core genetic algorithm engine for strategy evolution."""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.population: List[DecisionGenome] = []
        self.generation_stats: List[GenerationStats] = []
        self.current_generation = 0
        self.best_genome: Optional[DecisionGenome] = None
        self.convergence_counter = 0
        self.last_best_fitness = 0.0
        
        # Evolution callbacks
        self.fitness_evaluator: Optional[Callable] = None
        self.selection_method: Optional[Callable] = None
        self.crossover_method: Optional[Callable] = None
        self.mutation_method: Optional[Callable] = None
        
        logger.info(f"Genetic Engine initialized with population size {self.config.population_size}")
        
    async def initialize_population(self, initial_genomes: Optional[List[DecisionGenome]] = None):
        """Initialize the population with genomes."""
        try:
            if initial_genomes:
                self.population = initial_genomes[:self.config.population_size]
                logger.info(f"Initialized population with {len(self.population)} provided genomes")
            else:
                # Generate random population
                self.population = []
                for i in range(self.config.population_size):
                    genome = self._create_random_genome(f"genome_{i:03d}")
                    self.population.append(genome)
                logger.info(f"Generated random population of {len(self.population)} genomes")
                
            # Validate all genomes
            valid_genomes = [g for g in self.population if g.validate()]
            if len(valid_genomes) != len(self.population):
                logger.warning(f"Some genomes failed validation: {len(self.population) - len(valid_genomes)} invalid")
                self.population = valid_genomes
                
            # Emit population initialized event
            await event_bus.publish('evolution.population.initialized', {
                'population_size': len(self.population),
                'generation': self.current_generation
            })
            
        except Exception as e:
            raise EvolutionException(f"Error initializing population: {e}")
            
    async def evolve(self, max_generations: Optional[int] = None) -> DecisionGenome:
        """Run the genetic evolution process."""
        try:
            max_gen = max_generations or self.config.max_generations
            
            logger.info(f"Starting evolution for {max_gen} generations")
            
            for generation in range(max_gen):
                self.current_generation = generation
                
                # Evaluate fitness for current population
                await self._evaluate_population()
                
                # Record generation statistics
                stats = self._calculate_generation_stats()
                self.generation_stats.append(stats)
                
                # Check for convergence
                if self._check_convergence():
                    logger.info(f"Evolution converged at generation {generation}")
                    break
                    
                # Check fitness threshold
                if self.best_genome and self.best_genome.fitness_score >= self.config.fitness_threshold:
                    logger.info(f"Fitness threshold reached at generation {generation}")
                    break
                    
                # Create next generation
                await self._create_next_generation()
                
                # Log progress
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}: Best fitness = {self.best_genome.fitness_score:.4f}")
                    
            # Return best genome
            if self.best_genome:
                logger.info(f"Evolution completed. Best fitness: {self.best_genome.fitness_score:.4f}")
                return self.best_genome
            else:
                raise EvolutionException("No valid genome found after evolution")
                
        except Exception as e:
            raise EvolutionException(f"Error during evolution: {e}")
            
    async def _evaluate_population(self):
        """Evaluate fitness for all genomes in population."""
        if not self.fitness_evaluator:
            raise EvolutionException("Fitness evaluator not set")
            
        evaluation_tasks = []
        for genome in self.population:
            task = self._evaluate_genome_fitness(genome)
            evaluation_tasks.append(task)
            
        # Run evaluations concurrently
        await asyncio.gather(*evaluation_tasks)
        
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # Update best genome
        if self.population and (not self.best_genome or 
                               self.population[0].fitness_score > self.best_genome.fitness_score):
            self.best_genome = self.population[0]
            
    async def _evaluate_genome_fitness(self, genome: DecisionGenome):
        """Evaluate fitness for a single genome."""
        try:
            # Call fitness evaluator
            fitness_score = await self.fitness_evaluator(genome)
            genome.fitness_score = fitness_score
            genome.generation = self.current_generation
            
            # Emit fitness evaluation event
            await event_bus.publish('evolution.fitness.evaluated', {
                'genome_id': genome.genome_id,
                'fitness_score': fitness_score,
                'generation': self.current_generation
            })
            
        except Exception as e:
            logger.error(f"Error evaluating genome {genome.genome_id}: {e}")
            genome.fitness_score = 0.0
            
    async def _create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        try:
            new_population = []
            
            # Elitism: Keep best genomes
            elite_count = min(self.config.elite_size, len(self.population))
            elite_genomes = self.population[:elite_count]
            new_population.extend(elite_genomes)
            
            # Generate remaining population
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    child = await self._crossover_genomes(parent1, parent2)
                else:
                    child = parent1
                    
                # Mutation
                if random.random() < self.config.mutation_rate:
                    child = self._mutate_genome(child)
                    
                # Validate child
                if child.validate():
                    new_population.append(child)
                    
            # Update population
            self.population = new_population[:self.config.population_size]
            
            # Emit generation created event
            await event_bus.publish('evolution.generation.created', {
                'generation': self.current_generation + 1,
                'population_size': len(self.population),
                'elite_count': elite_count
            })
            
        except Exception as e:
            raise EvolutionException(f"Error creating next generation: {e}")
            
    def _select_parent(self) -> DecisionGenome:
        """Select a parent genome using tournament selection."""
        if not self.selection_method:
            return self._tournament_selection()
        else:
            return self.selection_method(self.population)
            
    def _tournament_selection(self) -> DecisionGenome:
        """Tournament selection method."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda g: g.fitness_score)
        
    async def _crossover_genomes(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Perform crossover between two parent genomes."""
        if not self.crossover_method:
            return self._uniform_crossover(parent1, parent2)
        else:
            return await self.crossover_method(parent1, parent2)
            
    def _uniform_crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Uniform crossover between genomes."""
        # Create child genome
        child = DecisionGenome(
            genome_id=f"child_{datetime.now().timestamp()}",
            parent_ids=[parent1.genome_id, parent2.genome_id],
            generation=self.current_generation + 1,
            crossover_count=1
        )
        
        # Crossover strategy parameters
        if random.random() < 0.5:
            child.strategy = parent1.strategy
        else:
            child.strategy = parent2.strategy
            
        # Crossover risk parameters
        if random.random() < 0.5:
            child.risk = parent1.risk
        else:
            child.risk = parent2.risk
            
        # Crossover timing parameters
        if random.random() < 0.5:
            child.timing = parent1.timing
        else:
            child.timing = parent2.timing
            
        # Crossover sensory weights (blend)
        child.sensory.price_weight = (parent1.sensory.price_weight + parent2.sensory.price_weight) / 2
        child.sensory.volume_weight = (parent1.sensory.volume_weight + parent2.sensory.volume_weight) / 2
        child.sensory.orderbook_weight = (parent1.sensory.orderbook_weight + parent2.sensory.orderbook_weight) / 2
        child.sensory.news_weight = (parent1.sensory.news_weight + parent2.sensory.news_weight) / 2
        child.sensory.sentiment_weight = (parent1.sensory.sentiment_weight + parent2.sensory.sentiment_weight) / 2
        child.sensory.economic_weight = (parent1.sensory.economic_weight + parent2.sensory.economic_weight) / 2
        
        # Crossover thinking weights (blend)
        child.thinking.trend_analysis_weight = (parent1.thinking.trend_analysis_weight + parent2.thinking.trend_analysis_weight) / 2
        child.thinking.risk_analysis_weight = (parent1.thinking.risk_analysis_weight + parent2.thinking.risk_analysis_weight) / 2
        child.thinking.performance_analysis_weight = (parent1.thinking.performance_analysis_weight + parent2.thinking.performance_analysis_weight) / 2
        child.thinking.pattern_recognition_weight = (parent1.thinking.pattern_recognition_weight + parent2.thinking.pattern_recognition_weight) / 2
        
        # Normalize weights
        child._normalize_weights()
        
        return child
        
    def _mutate_genome(self, genome: DecisionGenome) -> DecisionGenome:
        """Mutate a genome."""
        if not self.mutation_method:
            return genome.mutate(self.config.mutation_rate)
        else:
            return self.mutation_method(genome)
            
    def _calculate_generation_stats(self) -> GenerationStats:
        """Calculate statistics for current generation."""
        if not self.population:
            return GenerationStats(
                generation=self.current_generation,
                timestamp=datetime.now(),
                population_size=0,
                best_fitness=0.0,
                average_fitness=0.0,
                worst_fitness=0.0,
                diversity_score=0.0,
                mutation_count=0,
                crossover_count=0,
                elite_count=0
            )
            
        fitness_scores = [g.fitness_score for g in self.population]
        
        return GenerationStats(
            generation=self.current_generation,
            timestamp=datetime.now(),
            population_size=len(self.population),
            best_fitness=max(fitness_scores),
            average_fitness=sum(fitness_scores) / len(fitness_scores),
            worst_fitness=min(fitness_scores),
            diversity_score=self._calculate_diversity(),
            mutation_count=sum(g.mutation_count for g in self.population),
            crossover_count=sum(g.crossover_count for g in self.population),
            elite_count=self.config.elite_size
        )
        
    def _calculate_diversity(self) -> float:
        """Calculate population diversity score."""
        if len(self.population) < 2:
            return 0.0
            
        # Calculate diversity based on fitness variance
        fitness_scores = [g.fitness_score for g in self.population]
        variance = sum((score - sum(fitness_scores) / len(fitness_scores)) ** 2 for score in fitness_scores) / len(fitness_scores)
        
        return min(variance, 1.0)  # Normalize to [0, 1]
        
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if not self.best_genome:
            return False
            
        current_fitness = self.best_genome.fitness_score
        
        if abs(current_fitness - self.last_best_fitness) < 0.001:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
            
        self.last_best_fitness = current_fitness
        
        return self.convergence_counter >= self.config.convergence_patience
        
    def _create_random_genome(self, genome_id: str) -> DecisionGenome:
        """Create a random genome for initial population."""
        genome = DecisionGenome(genome_id=genome_id)
        
        # Randomize strategy parameters
        genome.strategy.entry_threshold = random.uniform(0.3, 0.7)
        genome.strategy.exit_threshold = random.uniform(0.3, 0.7)
        genome.strategy.momentum_weight = random.uniform(0.1, 0.5)
        genome.strategy.trend_weight = random.uniform(0.2, 0.6)
        genome.strategy.volume_weight = random.uniform(0.1, 0.4)
        genome.strategy.sentiment_weight = random.uniform(0.05, 0.2)
        
        # Randomize risk parameters
        genome.risk.risk_tolerance = random.uniform(0.2, 0.8)
        genome.risk.position_size_multiplier = random.uniform(0.5, 2.0)
        genome.risk.stop_loss_threshold = random.uniform(0.01, 0.05)
        genome.risk.take_profit_threshold = random.uniform(0.02, 0.08)
        
        # Randomize timing parameters
        genome.timing.holding_period_min = random.randint(1, 10)
        genome.timing.holding_period_max = random.randint(20, 60)
        
        # Randomize sensory weights
        weights = [random.random() for _ in range(6)]
        total = sum(weights)
        genome.sensory.price_weight = weights[0] / total
        genome.sensory.volume_weight = weights[1] / total
        genome.sensory.orderbook_weight = weights[2] / total
        genome.sensory.news_weight = weights[3] / total
        genome.sensory.sentiment_weight = weights[4] / total
        genome.sensory.economic_weight = weights[5] / total
        
        # Randomize thinking weights
        weights = [random.random() for _ in range(4)]
        total = sum(weights)
        genome.thinking.trend_analysis_weight = weights[0] / total
        genome.thinking.risk_analysis_weight = weights[1] / total
        genome.thinking.performance_analysis_weight = weights[2] / total
        genome.thinking.pattern_recognition_weight = weights[3] / total
        
        return genome
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        return {
            'total_generations': self.current_generation,
            'population_size': len(self.population),
            'best_fitness': self.best_genome.fitness_score if self.best_genome else 0.0,
            'best_genome_id': self.best_genome.genome_id if self.best_genome else None,
            'generation_stats': [stats.__dict__ for stats in self.generation_stats],
            'config': {
                'population_size': self.config.population_size,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size
            }
        } 