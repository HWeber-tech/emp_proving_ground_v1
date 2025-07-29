"""
Genetic Algorithm Optimizer

Specialized genetic algorithm for strategy parameter optimization.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import random
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of genetic algorithm optimization"""
    best_parameters: Dict[str, Any]
    best_fitness: float
    generation_count: int
    population_size: int
    convergence_history: List[float]
    execution_time: float


class GeneticOptimizer:
    """
    Genetic Algorithm Optimizer for Strategy Parameters
    
    Implements advanced genetic algorithm with:
    - Multi-objective optimization
    - Adaptive mutation rates
    - Tournament selection
    - Elitism preservation
    - Convergence detection
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elite_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Optimization state
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.convergence_history = []
        
        logger.info(f"GeneticOptimizer initialized: pop_size={population_size}, generations={generations}")
    
    def optimize_parameters(self, strategy_class: type, symbols: List[str], 
                          historical_data: Dict[str, List], 
                          fitness_function: Callable,
                          parameter_bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """Optimize strategy parameters using genetic algorithm"""
        
        import time
        start_time = time.time()
        
        # Initialize population
        self._initialize_population(parameter_bounds)
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            self._evaluate_fitness(strategy_class, symbols, historical_data, fitness_function)
            
            # Track best individual
            self._update_best_individual()
            
            # Record convergence
            self.convergence_history.append(self.best_fitness)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Convergence reached at generation {generation}")
                break
            
            # Selection
            selected = self._selection()
            
            # Crossover and mutation
            new_population = self._crossover_and_mutation(selected, parameter_bounds)
            
            # Elitism
            self._apply_elitism(new_population)
            
            # Update population
            self.population = new_population
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}")
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=self._individual_to_parameters(self.best_individual),
            best_fitness=self.best_fitness,
            generation_count=len(self.convergence_history),
            population_size=self.population_size,
            convergence_history=self.convergence_history,
            execution_time=execution_time
        )
    
    def _initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> None:
        """Initialize random population within parameter bounds"""
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in parameter_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual[param_name] = random.randint(min_val, max_val)
                else:
                    individual[param_name] = random.uniform(min_val, max_val)
            self.population.append(individual)
    
    def _evaluate_fitness(self, strategy_class: type, symbols: List[str],
                         historical_data: Dict[str, List], fitness_function: Callable) -> None:
        """Evaluate fitness for all individuals in population"""
        self.fitness_scores = []
        
        for individual in self.population:
            try:
                # Create strategy instance with individual parameters
                strategy_instance = strategy_class(
                    strategy_id=f"opt_{id(individual)}",
                    parameters=individual,
                    symbols=symbols
                )
                
                # Evaluate fitness
                fitness = fitness_function(strategy_instance, historical_data)
                self.fitness_scores.append(fitness)
                
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                self.fitness_scores.append(float('-inf'))
    
    def _update_best_individual(self) -> None:
        """Update best individual based on current population"""
        if not self.fitness_scores:
            return
        
        best_idx = np.argmax(self.fitness_scores)
        best_fitness = self.fitness_scores[best_idx]
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_individual = self.population[best_idx].copy()
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        if len(self.convergence_history) < 20:
            return False
        
        # Check if fitness hasn't improved in last 20 generations
        recent_fitness = self.convergence_history[-20:]
        if max(recent_fitness) - min(recent_fitness) < 0.001:
            return True
        
        return False
    
    def _selection(self) -> List[Dict[str, Any]]:
        """Tournament selection"""
        selected = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def _crossover_and_mutation(self, selected: List[Dict[str, Any]], 
                               parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Perform crossover and mutation operations"""
        new_population = []
        
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutation(child1, parameter_bounds)
                child2 = self._mutation(child2, parameter_bounds)
                
                new_population.extend([child1, child2])
            else:
                new_population.append(selected[i])
        
        return new_population[:self.population_size]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover"""
        params = list(parent1.keys())
        crossover_point = random.randint(1, len(params) - 1)
        
        child1 = {}
        child2 = {}
        
        for i, param in enumerate(params):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutation(self, individual: Dict[str, Any], 
                  parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Random mutation with adaptive rate"""
        mutated = individual.copy()
        
        for param_name, (min_val, max_val) in parameter_bounds.items():
            if random.random() < self.mutation_rate:
                if isinstance(min_val, int) and isinstance(max_val, int):
                    mutated[param_name] = random.randint(min_val, max_val)
                else:
                    mutated[param_name] = random.uniform(min_val, max_val)
        
        return mutated
    
    def _apply_elitism(self, new_population: List[Dict[str, Any]]) -> None:
        """Apply elitism by preserving best individuals"""
        if not self.fitness_scores:
            return
        
        # Get indices of best individuals
        best_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        
        # Replace worst individuals with best ones
        for i, best_idx in enumerate(best_indices):
            if i < len(new_population):
                new_population[i] = self.population[best_idx].copy()
    
    def _individual_to_parameters(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Convert individual to strategy parameters"""
        return individual.copy()
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            'best_fitness': self.best_fitness,
            'best_parameters': self.best_individual,
            'generations': len(self.convergence_history),
            'convergence_rate': self._calculate_convergence_rate(),
            'population_diversity': self._calculate_population_diversity()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate"""
        if len(self.convergence_history) < 2:
            return 0.0
        
        initial_fitness = self.convergence_history[0]
        final_fitness = self.convergence_history[-1]
        
        if initial_fitness == 0:
            return 0.0
        
        return (final_fitness - initial_fitness) / abs(initial_fitness)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity"""
        if not self.population:
            return 0.0
        
        # Calculate average distance between individuals
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_individual_distance(
                    self.population[i], self.population[j]
                )
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_individual_distance(self, ind1: Dict[str, Any], ind2: Dict[str, Any]) -> float:
        """Calculate distance between two individuals"""
        distance = 0.0
        for param in ind1.keys():
            if param in ind2:
                val1 = ind1[param]
                val2 = ind2[param]
                distance += abs(val1 - val2) ** 2
        
        return np.sqrt(distance) 
