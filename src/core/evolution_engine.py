"""
Real Evolution Engine Implementation
Replaces the stub with functional genetic programming
"""

import asyncio
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import uuid
from concurrent.futures import ProcessPoolExecutor
import hashlib

# Import DecisionGenome from the top-level genome package rather than via the
# ``src`` namespace. When ``src`` is added to ``sys.path`` (as in the CI
# import test), attempting to import via ``src.genome`` will resolve to
# ``src/src/genome``, which does not exist and can trigger
# "attempted relative import beyond top-level package" errors.  Importing
# directly from the ``genome`` package avoids this issue and still points to
# the same module because ``genome`` is a top-level package within ``src``.
from genome.models.genome import DecisionGenome
from ..config.evolution_config import EvolutionConfig

logger = logging.getLogger(__name__)

@dataclass
class EvolutionStats:
    """Statistics for evolution tracking"""
    generation: int
    best_fitness: float
    average_fitness: float
    diversity_index: float
    convergence_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PopulationMember:
    """Individual in the population"""
    genome: DecisionGenome
    fitness: float = 0.0
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)

class RealEvolutionEngine:
    """
    Real implementation of the Evolution Engine.
    Replaces the stub with functional genetic programming.
    """
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.current_generation = 0
        self.population: List[PopulationMember] = []
        self.evolution_history: List[EvolutionStats] = []
        self.best_genome: Optional[DecisionGenome] = None
        self.is_evolving = False
        
        # Performance tracking
        self.fitness_cache: Dict[str, float] = {}
        self.diversity_tracker = DiversityTracker()
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"RealEvolutionEngine initialized with population size: {config.population_size}")
    
    def _initialize_population(self) -> None:
        """Initialize population with random genomes"""
        self.population = []
        for i in range(self.config.population_size):
            genome = self._create_random_genome()
            member = PopulationMember(
                genome=genome,
                fitness=0.0,
                age=0,
                parent_ids=[]
            )
            self.population.append(member)
    
    def _create_random_genome(self) -> DecisionGenome:
        """Create a random genome"""
        return DecisionGenome(
            genome_id=str(uuid.uuid4()),
            version="1.1.0"
        )
    
    def get_population(self) -> List[DecisionGenome]:
        """Get the current population of genomes"""
        return [member.genome for member in self.population]
    
    async def evolve_generation(self) -> None:
        """Evolve to the next generation using genetic programming"""
        if self.is_evolving:
            logger.warning("Evolution already in progress, skipping")
            return
        
        self.is_evolving = True
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting evolution generation {self.current_generation + 1}")
            
            # Step 1: Evaluate fitness of current population
            await self._evaluate_population_fitness()
            
            # Step 2: Update best genome
            best_member = max(self.population, key=lambda x: x.fitness)
            if self.best_genome is None or best_member.fitness > (getattr(self.best_genome, 'fitness_score', 0) or 0):
                self.best_genome = self._copy_genome(best_member.genome)
                self.best_genome.fitness_score = best_member.fitness
                logger.info(f"New best genome found with fitness: {best_member.fitness:.4f}")
            
            # Step 3: Selection
            parents = self._select_parents()
            
            # Step 4: Create offspring through crossover and mutation
            offspring = self._create_offspring(parents)
            
            # Step 5: Replacement strategy
            self._replace_population(offspring)
            
            # Step 6: Track statistics
            stats = self._calculate_generation_stats()
            self.evolution_history.append(stats)
            
            self.current_generation += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generation {self.current_generation} completed in {duration:.2f}s")
            logger.info(f"Best fitness: {stats.best_fitness:.4f}, Average: {stats.average_fitness:.4f}")
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
            raise
        finally:
            self.is_evolving = False
    
    async def _evaluate_population_fitness(self) -> None:
        """Evaluate fitness for entire population"""
        if self.config.parallel_evaluation:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                fitness_futures = [
                    executor.submit(self._evaluate_genome_sync, member.genome) 
                    for member in self.population
                ]
                fitness_scores = [future.result() for future in fitness_futures]
        else:
            fitness_scores = []
            for member in self.population:
                fitness = await self.evaluate_fitness(member.genome)
                fitness_scores.append(fitness)
        
        # Update fitness values
        for member, fitness in zip(self.population, fitness_scores):
            member.fitness = fitness
    
    def _evaluate_genome_sync(self, genome: DecisionGenome) -> float:
        """Synchronous wrapper for parallel fitness evaluation"""
        import asyncio
        return asyncio.run(self.evaluate_fitness(genome))
    
    async def evaluate_fitness(self, genome: DecisionGenome) -> float:
        """Evaluate fitness of a single genome"""
        # Create cache key
        cache_key = self._create_cache_key(genome)
        
        # Check cache first
        if cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]
        
        # Evaluate fitness based on genome performance
        fitness = await self._calculate_genome_fitness(genome)
        
        # Cache result
        self.fitness_cache[cache_key] = fitness
        
        return fitness
    
    async def _calculate_genome_fitness(self, genome: DecisionGenome) -> float:
        """Calculate fitness score for a genome"""
        # This is a simplified fitness calculation
        # In practice, this would involve backtesting the strategy
        try:
            # Calculate fitness based on genome parameters
            
            # Strategy fitness component
            strategy_score = (
                genome.strategy.entry_threshold * 0.3 +
                genome.strategy.exit_threshold * 0.3 +
                (genome.strategy.momentum_weight + genome.strategy.trend_weight) * 0.2 +
                (1.0 / max(genome.strategy.lookback_period, 1)) * 0.2
            )
            
            # Risk fitness component
            risk_score = (
                genome.risk.risk_tolerance * 0.4 +
                (1.0 - genome.risk.stop_loss_threshold) * 0.3 +
                (1.0 - genome.risk.max_drawdown_limit) * 0.3
            )
            
            # Timing fitness component
            timing_score = (
                (genome.timing.holding_period_max - genome.timing.holding_period_min) / 30.0 * 0.5 +
                (1.0 / max(genome.timing.reentry_delay, 1)) * 0.5
            )
            
            # Combined fitness
            fitness = (strategy_score + risk_score + timing_score) / 3.0
            
            # Ensure reasonable bounds
            fitness = max(0.0, min(1.0, fitness))
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0
    
    def _create_cache_key(self, genome: DecisionGenome) -> str:
        """Create cache key for genome"""
        genome_str = json.dumps(genome.to_dict(), sort_keys=True)
        return hashlib.md5(genome_str.encode()).hexdigest()
    
    def _select_parents(self) -> List[PopulationMember]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(self.config.num_parents):
            # Tournament selection
            tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _create_offspring(self, parents: List[PopulationMember]) -> List[PopulationMember]:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_genome, child2_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                child1_genome = self._copy_genome(parent1.genome)
                child2_genome = self._copy_genome(parent2.genome)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1_genome = self._mutate(child1_genome)
            if random.random() < self.config.mutation_rate:
                child2_genome = self._mutate(child2_genome)
            
            # Create offspring members
            child1 = PopulationMember(
                genome=child1_genome,
                fitness=0.0,
                age=0,
                parent_ids=[parent1.genome.genome_id, parent2.genome.genome_id]
            )
            
            child2 = PopulationMember(
                genome=child2_genome,
                fitness=0.0,
                age=0,
                parent_ids=[parent1.genome.genome_id, parent2.genome.genome_id]
            )
            
            offspring.extend([child1, child2])
        
        return offspring[:self.config.offspring_size]
    
    def _copy_genome(self, genome: DecisionGenome) -> DecisionGenome:
        """Create a deep copy of a genome"""
        return DecisionGenome.from_dict(genome.to_dict())
    
    def _crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> Tuple[DecisionGenome, DecisionGenome]:
        """Perform crossover between two parents"""
        child1 = self._copy_genome(parent1)
        child2 = self._copy_genome(parent2)
        
        # Crossover strategy parameters
        if random.random() < 0.5:
            child1.strategy, child2.strategy = child2.strategy, child1.strategy
        
        if random.random() < 0.5:
            child1.risk, child2.risk = child2.risk, child1.risk
        
        if random.random() < 0.5:
            child1.timing, child2.timing = child2.timing, child1.timing
        
        # Update genome IDs
        child1.genome_id = str(uuid.uuid4())
        child2.genome_id = str(uuid.uuid4())
        
        return child1, child2
    
    def _mutate(self, genome: DecisionGenome) -> DecisionGenome:
        """Perform mutation on a genome"""
        mutated = self._copy_genome(genome)
        mutated.genome_id = str(uuid.uuid4())
        mutated.parent_ids = [genome.genome_id]
        mutated.mutation_count = genome.mutation_count + 1
        mutated.generation = genome.generation + 1
        
        # Mutate strategy parameters
        if random.random() < 0.1:
            mutated.strategy.entry_threshold += random.uniform(-0.1, 0.1)
            mutated.strategy.entry_threshold = max(0, min(1, mutated.strategy.entry_threshold))
            
        if random.random() < 0.1:
            mutated.strategy.exit_threshold += random.uniform(-0.1, 0.1)
            mutated.strategy.exit_threshold = max(0, min(1, mutated.strategy.exit_threshold))
            
        # Mutate risk parameters
        if random.random() < 0.1:
            mutated.risk.risk_tolerance += random.uniform(-0.1, 0.1)
            mutated.risk.risk_tolerance = max(0, min(1, mutated.risk.risk_tolerance))
            
        # Mutate timing parameters
        if random.random() < 0.1:
            mutated.timing.holding_period_min += random.randint(-2, 2)
            mutated.timing.holding_period_min = max(0, mutated.timing.holding_period_min)
            
        # Ensure weights are normalized
        mutated.validate()
        
        return mutated
    
    def _replace_population(self, offspring: List[PopulationMember]) -> None:
        """Replace population using elitist strategy"""
        # Combine current population and offspring
        combined = self.population + offspring
        
        # Sort by fitness (descending)
        combined.sort(key=lambda x: x.fitness, reverse=True)
        
        # Select top individuals for next generation
        self.population = combined[:self.config.population_size]
        
        # Increment age for surviving members
        for member in self.population:
            member.age += 1
    
    def _calculate_generation_stats(self) -> EvolutionStats:
        """Calculate statistics for the current generation"""
        if not self.population:
            return EvolutionStats(
                generation=self.current_generation,
                best_fitness=0.0,
                average_fitness=0.0,
                diversity_index=0.0,
                convergence_rate=0.0
            )
        
        fitness_scores = [member.fitness for member in self.population]
        
        best_fitness = float(max(fitness_scores))
        average_fitness = float(np.mean(fitness_scores))
        diversity_index = float(self.diversity_tracker.calculate_diversity([m.genome for m in self.population]))
        
        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.evolution_history) >= 2:
            recent_best = [stats.best_fitness for stats in self.evolution_history[-5:]]
            if len(recent_best) >= 2:
                improvements = [recent_best[i] - recent_best[i-1] for i in range(1, len(recent_best))]
                convergence_rate = float(np.mean(improvements))
        
        return EvolutionStats(
            generation=self.current_generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            diversity_index=diversity_index,
            convergence_rate=convergence_rate
        )
    
    def get_evolution_stats(self) -> List[EvolutionStats]:
        """Get evolution history statistics"""
        return self.evolution_history.copy()
    
    def save_state(self, filepath: str) -> None:
        """Save evolution engine state to file"""
        state = {
            'current_generation': self.current_generation,
            'evolution_history': [vars(stats) for stats in self.evolution_history],
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'population': [
                {
                    'genome': member.genome.to_dict(),
                    'fitness': member.fitness,
                    'age': member.age,
                    'parent_ids': member.parent_ids
                }
                for member in self.population
            ],
            'config': vars(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Evolution state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load evolution engine state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_generation = state['current_generation']
        self.evolution_history = [EvolutionStats(**stats) for stats in state['evolution_history']]
        
        if state['best_genome']:
            self.best_genome = DecisionGenome.from_dict(state['best_genome'])
        
        self.population = []
        for member_data in state['population']:
            genome = DecisionGenome.from_dict(member_data['genome'])
            member = PopulationMember(
                genome=genome,
                fitness=member_data['fitness'],
                age=member_data['age'],
                parent_ids=member_data['parent_ids']
            )
            self.population.append(member)
        
        logger.info(f"Evolution state loaded from {filepath}")

class DiversityTracker:
    """Track genetic diversity in the population"""
    
    def calculate_diversity(self, genomes: List[DecisionGenome]) -> float:
        """Calculate diversity index for the population"""
        if len(genomes) < 2:
            return 0.0
        
        # Calculate pairwise distances based on key parameters
        distances = []
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distance = self._calculate_genome_distance(genomes[i], genomes[j])
                distances.append(distance)
        
        # Return average distance as diversity measure
        return float(np.mean(distances)) if distances else 0.0
    
    def _calculate_genome_distance(self, genome1: DecisionGenome, genome2: DecisionGenome) -> float:
        """Calculate distance between two genomes"""
        # Calculate distance based on key parameters
        strategy_params = [
            genome1.strategy.entry_threshold, genome1.strategy.exit_threshold,
            genome1.strategy.momentum_weight, genome1.strategy.trend_weight,
            genome2.strategy.entry_threshold, genome2.strategy.exit_threshold,
            genome2.strategy.momentum_weight, genome2.strategy.trend_weight
        ]
        
        risk_params = [
            genome1.risk.risk_tolerance, genome1.risk.stop_loss_threshold,
            genome1.risk.take_profit_threshold, genome1.risk.max_drawdown_limit,
            genome2.risk.risk_tolerance, genome2.risk.stop_loss_threshold,
            genome2.risk.take_profit_threshold, genome2.risk.max_drawdown_limit
        ]
        
        # Euclidean distance
        distance = 0.0
        for i in range(0, len(strategy_params), 2):
            distance += (strategy_params[i] - strategy_params[i+1]) ** 2
        
        for i in range(0, len(risk_params), 2):
            distance += (risk_params[i] - risk_params[i+1]) ** 2
        
        return float(np.sqrt(distance) / 4.0)  # Normalize
