"""
Evolution Configuration
Configuration for the genetic algorithm parameters
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine"""
    
    # Population parameters
    population_size: int = 50
    num_parents: int = 20
    offspring_size: int = 30
    
    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    
    # Selection parameters
    tournament_size: int = 3
    
    # Performance parameters
    parallel_evaluation: bool = True
    max_workers: int = 4
    
    # Evolution control
    max_generations: int = 1000
    convergence_threshold: float = 0.001
    
    # Logging
    log_level: str = "INFO"
    save_checkpoint_every: int = 10
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.crossover_rate < 0 or self.crossover_rate > 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.mutation_rate < 0 or self.mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if self.tournament_size <= 0:
            raise ValueError("Tournament size must be positive")
        return True
