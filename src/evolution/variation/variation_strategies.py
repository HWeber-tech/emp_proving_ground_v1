"""
EMP Variation Strategies v1.1

Modular variation strategies for the adaptive core.
Separates crossover and mutation logic from the main evolution engine.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from ...genome.models.genome import StrategyGenome
from ...core.events import FitnessReport

logger = logging.getLogger(__name__)


class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.crossover_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """Perform crossover between two parent genomes."""
        pass
        
    def log_crossover(self, parent1_id: str, parent2_id: str, 
                     child1_id: str, child2_id: str, generation: int):
        """Log crossover event."""
        entry = {
            "timestamp": datetime.now(),
            "strategy": self.name,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
            "child1_id": child1_id,
            "child2_id": child2_id,
            "generation": generation
        }
        self.crossover_history.append(entry)
        logger.debug(f"Crossover logged: {self.name} created {child1_id}, {child2_id}")


class UniformCrossover(CrossoverStrategy):
    """Uniform crossover strategy."""
    
    def __init__(self, crossover_rate: float = 0.5):
        super().__init__("uniform_crossover")
        self.crossover_rate = crossover_rate
        
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """Perform uniform crossover."""
        # Create copies of parents
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Generate crossover mask
        mask = np.random.random(len(parent1.parameters)) < self.crossover_rate
        
        # Apply crossover
        for i, should_cross in enumerate(mask):
            if should_cross:
                child1.parameters[i], child2.parameters[i] = child2.parameters[i], child1.parameters[i]
                
        # Update child IDs
        child1.id = f"{parent1.id}_{parent2.id}_child1"
        child2.id = f"{parent1.id}_{parent2.id}_child2"
        
        return child1, child2


class SinglePointCrossover(CrossoverStrategy):
    """Single-point crossover strategy."""
    
    def __init__(self):
        super().__init__("single_point_crossover")
        
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """Perform single-point crossover."""
        # Create copies of parents
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Choose crossover point
        crossover_point = random.randint(1, len(parent1.parameters) - 1)
        
        # Apply crossover
        child1.parameters[:crossover_point] = parent2.parameters[:crossover_point]
        child2.parameters[:crossover_point] = parent1.parameters[:crossover_point]
        
        # Update child IDs
        child1.id = f"{parent1.id}_{parent2.id}_child1"
        child2.id = f"{parent1.id}_{parent2.id}_child2"
        
        return child1, child2


class TwoPointCrossover(CrossoverStrategy):
    """Two-point crossover strategy."""
    
    def __init__(self):
        super().__init__("two_point_crossover")
        
    def crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """Perform two-point crossover."""
        # Create copies of parents
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Choose two crossover points
        points = sorted(random.sample(range(1, len(parent1.parameters)), 2))
        point1, point2 = points
        
        # Apply crossover
        child1.parameters[point1:point2] = parent2.parameters[point1:point2]
        child2.parameters[point1:point2] = parent1.parameters[point1:point2]
        
        # Update child IDs
        child1.id = f"{parent1.id}_{parent2.id}_child1"
        child2.id = f"{parent1.id}_{parent2.id}_child2"
        
        return child1, child2


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.mutation_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def mutate(self, genome: StrategyGenome, mutation_rate: float) -> StrategyGenome:
        """Perform mutation on a genome."""
        pass
        
    def log_mutation(self, parent_id: str, child_id: str, 
                    mutation_rate: float, generation: int):
        """Log mutation event."""
        entry = {
            "timestamp": datetime.now(),
            "strategy": self.name,
            "parent_id": parent_id,
            "child_id": child_id,
            "mutation_rate": mutation_rate,
            "generation": generation
        }
        self.mutation_history.append(entry)
        logger.debug(f"Mutation logged: {self.name} created {child_id}")


class GaussianMutation(MutationStrategy):
    """Gaussian mutation strategy."""
    
    def __init__(self, sigma: float = 0.1):
        super().__init__("gaussian_mutation")
        self.sigma = sigma
        
    def mutate(self, genome: StrategyGenome, mutation_rate: float) -> StrategyGenome:
        """Perform Gaussian mutation."""
        # Create copy of genome
        mutated = genome.copy()
        
        # Apply mutation to each parameter
        for i in range(len(mutated.parameters)):
            if random.random() < mutation_rate:
                # Add Gaussian noise
                noise = np.random.normal(0, self.sigma)
                mutated.parameters[i] += noise
                
                # Ensure parameter stays within bounds
                mutated.parameters[i] = np.clip(mutated.parameters[i], 0, 1)
                
        # Update ID
        mutated.id = f"{genome.id}_mutated"
        
        return mutated


class UniformMutation(MutationStrategy):
    """Uniform mutation strategy."""
    
    def __init__(self):
        super().__init__("uniform_mutation")
        
    def mutate(self, genome: StrategyGenome, mutation_rate: float) -> StrategyGenome:
        """Perform uniform mutation."""
        # Create copy of genome
        mutated = genome.copy()
        
        # Apply mutation to each parameter
        for i in range(len(mutated.parameters)):
            if random.random() < mutation_rate:
                # Replace with random value
                mutated.parameters[i] = random.random()
                
        # Update ID
        mutated.id = f"{genome.id}_mutated"
        
        return mutated


class AdaptiveMutation(MutationStrategy):
    """Adaptive mutation strategy based on fitness."""
    
    def __init__(self, base_sigma: float = 0.1, fitness_threshold: float = 0.5):
        super().__init__("adaptive_mutation")
        self.base_sigma = base_sigma
        self.fitness_threshold = fitness_threshold
        
    def mutate(self, genome: StrategyGenome, mutation_rate: float, 
              fitness_score: float = 0.5) -> StrategyGenome:
        """Perform adaptive mutation based on fitness."""
        # Create copy of genome
        mutated = genome.copy()
        
        # Adjust mutation strength based on fitness
        if fitness_score < self.fitness_threshold:
            # Low fitness: increase mutation strength
            adaptive_sigma = self.base_sigma * 2.0
        else:
            # High fitness: decrease mutation strength
            adaptive_sigma = self.base_sigma * 0.5
            
        # Apply mutation to each parameter
        for i in range(len(mutated.parameters)):
            if random.random() < mutation_rate:
                # Add adaptive Gaussian noise
                noise = np.random.normal(0, adaptive_sigma)
                mutated.parameters[i] += noise
                
                # Ensure parameter stays within bounds
                mutated.parameters[i] = np.clip(mutated.parameters[i], 0, 1)
                
        # Update ID
        mutated.id = f"{genome.id}_mutated"
        
        return mutated


class VariationFactory:
    """Factory for creating variation strategies."""
    
    _crossover_strategies = {
        "uniform": UniformCrossover,
        "single_point": SinglePointCrossover,
        "two_point": TwoPointCrossover
    }
    
    _mutation_strategies = {
        "gaussian": GaussianMutation,
        "uniform": UniformMutation,
        "adaptive": AdaptiveMutation
    }
    
    @classmethod
    def create_crossover_strategy(cls, strategy_name: str, **kwargs) -> CrossoverStrategy:
        """Create a crossover strategy by name."""
        if strategy_name not in cls._crossover_strategies:
            raise ValueError(f"Unknown crossover strategy: {strategy_name}")
            
        strategy_class = cls._crossover_strategies[strategy_name]
        return strategy_class(**kwargs)
        
    @classmethod
    def create_mutation_strategy(cls, strategy_name: str, **kwargs) -> MutationStrategy:
        """Create a mutation strategy by name."""
        if strategy_name not in cls._mutation_strategies:
            raise ValueError(f"Unknown mutation strategy: {strategy_name}")
            
        strategy_class = cls._mutation_strategies[strategy_name]
        return strategy_class(**kwargs)
        
    @classmethod
    def list_crossover_strategies(cls) -> List[str]:
        """List available crossover strategies."""
        return list(cls._crossover_strategies.keys())
        
    @classmethod
    def list_mutation_strategies(cls) -> List[str]:
        """List available mutation strategies."""
        return list(cls._mutation_strategies.keys())
        
    @classmethod
    def register_crossover_strategy(cls, name: str, strategy_class: type):
        """Register a new crossover strategy."""
        cls._crossover_strategies[name] = strategy_class
        logger.info(f"Registered crossover strategy: {name}")
        
    @classmethod
    def register_mutation_strategy(cls, name: str, strategy_class: type):
        """Register a new mutation strategy."""
        cls._mutation_strategies[name] = strategy_class
        logger.info(f"Registered mutation strategy: {name}") 