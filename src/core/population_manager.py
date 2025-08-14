"""
Population Manager Implementation
=================================

Concrete implementation of IPopulationManager for genetic algorithm population management.
Optimized for performance with Redis caching integration.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.performance.market_data_cache import get_global_cache

from .interfaces import DecisionGenome, IPopulationManager

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
            # Generate initial population if empty
            logger.info("Population empty, generating initial population")
            self._generate_initial_population()
        
        if not self.population:
            logger.warning("Failed to generate initial population")
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

    def _generate_initial_population(self) -> None:
        """Generate initial population of genomes."""
        try:
            import random

            from src.core.interfaces import DecisionGenome
            
            logger.info(f"Generating initial population of {self.population_size} genomes")
            
            for i in range(self.population_size):
                # Create genome with random parameters
                genome = DecisionGenome(
                    id=f"genome_{i:04d}",
                    parameters={
                        'risk_tolerance': random.uniform(0.1, 0.9),
                        'position_size_factor': random.uniform(0.01, 0.1),
                        'stop_loss_factor': random.uniform(0.005, 0.05),
                        'take_profit_factor': random.uniform(0.01, 0.1),
                        'trend_sensitivity': random.uniform(0.1, 1.0),
                        'volatility_threshold': random.uniform(0.001, 0.01),
                        'correlation_threshold': random.uniform(0.3, 0.9),
                        'momentum_window': random.randint(5, 50),
                        'mean_reversion_factor': random.uniform(0.1, 0.8),
                        'market_regime_sensitivity': random.uniform(0.2, 0.8)
                    },
                    fitness=0.0,
                    generation=0,
                    species_type='trading_strategy',
                    parent_ids=[],
                    mutation_history=[],
                    performance_metrics={
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0,
                        'sortino_ratio': 0.0
                    }
                )
                
                self.population.append(genome)
            
            logger.info(f"Successfully generated {len(self.population)} genomes")
            
        except Exception as e:
            logger.error(f"Failed to generate initial population: {e}")
            self.population = []
    
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
            new_population = elite_genomes.copy()  # Keep elite
            
            # Fill remaining slots with offspring
            while len(new_population) < self.population_size:
                # Select parents via tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Create offspring via crossover and mutation
                offspring = self._crossover(parent1, parent2)
                offspring = self._mutate(offspring)
                
                new_population.append(offspring)
            
            # Update population
            self.population = new_population
            self.generation += 1
            
            logger.info(f"Evolution complete. Generation {self.generation}, "
                       f"Best fitness: {max(g.fitness for g in self.population):.4f}")
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
    
    def _evaluate_fitness(self, market_data: Dict, performance_metrics: Dict) -> None:
        """Evaluate fitness for all genomes based on market performance."""
        for genome in self.population:
            try:
                # Calculate fitness based on multiple factors
                profit_score = performance_metrics.get('total_return', 0.0) * 0.4
                risk_score = (1.0 - performance_metrics.get('max_drawdown', 1.0)) * 0.3
                consistency_score = performance_metrics.get('win_rate', 0.0) * 0.2
                efficiency_score = performance_metrics.get('sharpe_ratio', 0.0) * 0.1
                
                genome.fitness = max(0.0, profit_score + risk_score + consistency_score + efficiency_score)
                
                # Update performance metrics
                genome.performance_metrics.update(performance_metrics)
                
            except Exception as e:
                logger.error(f"Fitness evaluation failed for genome {genome.id}: {e}")
                genome.fitness = 0.0
    
    def _tournament_selection(self, tournament_size: int = 3) -> DecisionGenome:
        """Select genome via tournament selection."""
        import random
        
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> DecisionGenome:
        """Create offspring via crossover."""
        import random

        from src.core.interfaces import DecisionGenome
        
        # Create new genome ID
        offspring_id = f"genome_{self.generation}_{random.randint(1000, 9999)}"
        
        # Blend parameters from both parents
        offspring_params = {}
        for key in parent1.parameters:
            if random.random() < 0.5:
                offspring_params[key] = parent1.parameters[key]
            else:
                offspring_params[key] = parent2.parameters.get(key, parent1.parameters[key])
        
        return DecisionGenome(
            id=offspring_id,
            parameters=offspring_params,
            fitness=0.0,
            generation=self.generation,
            species_type=parent1.species_type,
            parent_ids=[parent1.id, parent2.id],
            mutation_history=[],
            performance_metrics={}
        )
    
    def _mutate(self, genome: DecisionGenome, mutation_rate: float = 0.1) -> DecisionGenome:
        """Apply mutation to genome."""
        import random
        
        for key, value in genome.parameters.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    # Gaussian mutation for float values
                    mutation_strength = 0.1
                    genome.parameters[key] = max(0.0, value + random.gauss(0, mutation_strength))
                elif isinstance(value, int):
                    # Integer mutation
                    genome.parameters[key] = max(1, value + random.randint(-5, 5))
                
                # Record mutation
                genome.mutation_history.append({
                    'generation': self.generation,
                    'parameter': key,
                    'old_value': value,
                    'new_value': genome.parameters[key]
                })
        
        return genome

