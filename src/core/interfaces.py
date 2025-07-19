"""
EMP Core Interfaces v1.1

Core interfaces and abstract base classes for the EMP Ultimate Architecture.
Defines contracts for swappable components across all layers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime

from src.genome.models.genome import DecisionGenome


class SensoryOrgan(ABC):
    """
    Interface for sensory organs that process market data.
    
    Defines the contract for sensory components that analyze different
    aspects of market data and produce sensory signals.
    """
    
    @abstractmethod
    def process(self, market_data: Any) -> Any:
        """
        Process market data and return sensory signals.
        
        Args:
            market_data: Market data to process
            
        Returns:
            Sensory signals
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this sensory organ."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Return the weight/importance of this sensory organ."""
        pass


class ISelectionStrategy(ABC):
    """
    Interface for selection strategies in the evolutionary process.
    
    This interface defines the contract for selecting genomes from a population
    based on their fitness scores. Implementations can include tournament selection,
    roulette wheel, rank-based selection, etc.
    """
    
    @abstractmethod
    def select(self, population: List[DecisionGenome], fitness_scores: List[float]) -> DecisionGenome:
        """
        Select a genome from the population based on fitness scores.
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            
        Returns:
            Selected genome
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this selection strategy."""
        pass


class IPopulationManager(ABC):
    """
    Interface for managing the population of genomes in the evolutionary process.
    
    This interface encapsulates all population-related operations, including
    initialization, state management, and lifecycle tracking.
    """
    
    @abstractmethod
    def initialize_population(self, genome_factory) -> None:
        """
        Initialize the population with new genomes.
        
        Args:
            genome_factory: Callable that creates new genomes
        """
        pass
    
    @abstractmethod
    def get_population(self) -> List[DecisionGenome]:
        """Get the current population."""
        pass
    
    @abstractmethod
    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """
        Get the top N genomes by fitness.
        
        Args:
            count: Number of top genomes to return
            
        Returns:
            List of best genomes
        """
        pass
    
    @abstractmethod
    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """
        Replace the current population with a new one.
        
        Args:
            new_population: New population to set
        """
        pass
    
    @abstractmethod
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        pass
    
    @property
    @abstractmethod
    def population_size(self) -> int:
        """Return the configured population size."""
        pass
    
    @property
    @abstractmethod
    def generation(self) -> int:
        """Return the current generation number."""
        pass
    
    @abstractmethod
    def advance_generation(self) -> None:
        """Increment the generation counter."""
        pass


class ICrossoverStrategy(ABC):
    """
    Interface for crossover strategies in genetic algorithms.
    
    Defines how two parent genomes combine to create offspring.
    """
    
    @abstractmethod
    def crossover(self, parent1: DecisionGenome, parent2: DecisionGenome) -> tuple[DecisionGenome, DecisionGenome]:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of two child genomes
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this crossover strategy."""
        pass


class IMutationStrategy(ABC):
    """
    Interface for mutation strategies in genetic algorithms.
    
    Defines how genomes are mutated to introduce genetic diversity.
    """
    
    @abstractmethod
    def mutate(self, genome: DecisionGenome, mutation_rate: float) -> DecisionGenome:
        """
        Apply mutation to a genome.
        
        Args:
            genome: Genome to mutate
            mutation_rate: Probability of mutation for each gene
            
        Returns:
            Mutated genome
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this mutation strategy."""
        pass


class IFitnessEvaluator(ABC):
    """
    Interface for fitness evaluation in evolutionary algorithms.
    
    Defines how genomes are evaluated and assigned fitness scores.
    """
    
    @abstractmethod
    def evaluate(self, genome: DecisionGenome) -> float:
        """
        Evaluate a genome and return its fitness score.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this fitness evaluator."""
        pass


class IGenomeFactory(ABC):
    """
    Interface for creating new genomes.
    
    Defines how new genomes are generated, either randomly or based on templates.
    """
    
    @abstractmethod
    def create_genome(self) -> DecisionGenome:
        """Create a new genome."""
        pass
    
    @abstractmethod
    def create_from_template(self, template: Dict[str, Any]) -> DecisionGenome:
        """
        Create a genome from a template.
        
        Args:
            template: Template dictionary with genome parameters
            
        Returns:
            New genome based on template
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the population manager to initial state."""
        pass


class IEvolutionLogger(ABC):
    """
    Interface for logging evolution progress and statistics.
    """
    
    @abstractmethod
    def log_generation(self, generation: int, population_stats: Dict[str, Any]) -> None:
        """Log statistics for a generation."""
        pass
    
    @abstractmethod
    def log_best_genome(self, genome: DecisionGenome, fitness: float) -> None:
        """Log the best genome found."""
        pass
    
    @abstractmethod
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get the complete evolution history."""
        pass
