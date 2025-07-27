"""
Population Manager Implementation
===============================

Concrete implementation of the IPopulationManager interface for managing
populations in genetic algorithms.
"""

import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
from pathlib import Path

from src.core.interfaces import IPopulationManager, DecisionGenome
from src.core.exceptions import EvolutionException

logger = logging.getLogger(__name__)


class PopulationManager(IPopulationManager):
    """Concrete implementation of population management for genetic algorithms."""
    
    def __init__(self, population_size: int = 100, max_generations: int = 1000):
        """Initialize population manager.
        
        Args:
            population_size: Size of the population
            max_generations: Maximum number of generations
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.current_generation = 0
        self.population: List[DecisionGenome] = []
        self.fitness_scores: List[float] = []
        self.history: List[Dict[str, Any]] = []
        self.genome_factory: Optional[Callable] = None
        
    def initialize_population(self, genome_factory: Callable) -> None:
        """Initialize the population with new genomes.
        
        Args:
            genome_factory: Factory function to create new genomes
        """
        self.genome_factory = genome_factory
        self.population = []
        self.fitness_scores = []
        
        for i in range(self.population_size):
            genome = genome_factory()
            genome.generation = 0
            genome.id = f"genome_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.population.append(genome)
            self.fitness_scores.append(0.0)
            
        logger.info(f"Initialized population with {len(self.population)} genomes")
        
    def get_population(self) -> List[DecisionGenome]:
        """Get the current population."""
        return self.population
        
    def get_best_genomes(self, count: int) -> List[DecisionGenome]:
        """Get the top N genomes by fitness.
        
        Args:
            count: Number of top genomes to return
            
        Returns:
            List of top genomes sorted by fitness
        """
        if not self.population or not self.fitness_scores:
            return []
            
        # Pair genomes with fitness scores and sort
        paired = list(zip(self.population, self.fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # Return top genomes
        return [genome for genome, _ in paired[:count]]
        
    def update_population(self, new_population: List[DecisionGenome]) -> None:
        """Replace the current population with a new one.
        
        Args:
            new_population: New population to replace current one
        """
        if len(new_population) != self.population_size:
            raise EvolutionException(
                f"New population size {len(new_population)} does not match expected {self.population_size}"
            )
            
        self.population = new_population
        self.fitness_scores = [0.0] * len(new_population)
        
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population.
        
        Returns:
            Dictionary containing population statistics
        """
        if not self.fitness_scores:
            return {
                'population_size': len(self.population),
                'generation': self.current_generation,
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'min_fitness': 0.0,
                'std_fitness': 0.0,
                'diversity': 0.0
            }
            
        fitness_array = np.array(self.fitness_scores)
        
        # Calculate diversity based on parameter variance
        diversity = 0.0
        if self.population:
            # Simple diversity metric based on parameter uniqueness
            unique_params = set()
            for genome in self.population:
                param_str = str(sorted(genome.parameters.items()))
                unique_params.add(param_str)
            diversity = len(unique_params) / len(self.population)
        
        return {
            'population_size': len(self.population),
            'generation': self.current_generation,
            'avg_fitness': float(np.mean(fitness_array)),
            'max_fitness': float(np.max(fitness_array)),
            'min_fitness': float(np.min(fitness_array)),
            'std_fitness': float(np.std(fitness_array)),
            'diversity': diversity
        }
        
    def advance_generation(self) -> None:
        """Increment the generation counter."""
        self.current_generation += 1
        
        # Log generation advancement
        stats = self.get_population_statistics()
        self.history.append({
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'statistics': stats
        })
        
        logger.info(f"Advanced to generation {self.current_generation}")
        
    def reset(self) -> None:
        """Reset the population manager to initial state."""
        self.current_generation = 0
        self.population = []
        self.fitness_scores = []
        self.history = []
        logger.info("Population manager reset to initial state")
        
    def update_fitness_scores(self, scores: List[float]) -> None:
        """Update fitness scores for the current population.
        
        Args:
            scores: List of fitness scores matching population order
        """
        if len(scores) != len(self.population):
            raise EvolutionException(
                f"Fitness scores count {len(scores)} does not match population size {len(self.population)}"
            )
            
        self.fitness_scores = scores
        
    def save_state(self, filepath: str) -> None:
        """Save population state to file.
        
        Args:
            filepath: Path to save the state
        """
        state = {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'current_generation': self.current_generation,
            'population': [genome.model_dump() for genome in self.population],
            'fitness_scores': self.fitness_scores,
            'history': self.history
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Population state saved to {filepath}")
        
    def load_state(self, filepath: str) -> None:
        """Load population state from file.
        
        Args:
            filepath: Path to load the state from
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.population_size = state['population_size']
            self.max_generations = state['max_generations']
            self.current_generation = state['current_generation']
            self.population = [DecisionGenome(**genome_data) for genome_data in state['population']]
            self.fitness_scores = state['fitness_scores']
            self.history = state.get('history', [])
            
            logger.info(f"Population state loaded from {filepath}")
            
        except Exception as e:
            raise EvolutionException(f"Failed to load population state: {str(e)}")
            
    def get_genome_by_id(self, genome_id: str) -> Optional[DecisionGenome]:
        """Get a specific genome by ID.
        
        Args:
            genome_id: ID of the genome to find
            
        Returns:
            Genome if found, None otherwise
        """
        for genome in self.population:
            if genome.id == genome_id:
                return genome
        return None
        
    def remove_genome(self, genome_id: str) -> bool:
        """Remove a genome from the population.
        
        Args:
            genome_id: ID of the genome to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, genome in enumerate(self.population):
            if genome.id == genome_id:
                del self.population[i]
                del self.fitness_scores[i]
                return True
        return False
        
    def add_genome(self, genome: DecisionGenome) -> None:
        """Add a new genome to the population.
        
        Args:
            genome: Genome to add
        """
        if len(self.population) >= self.population_size:
            raise EvolutionException(
                f"Cannot add genome, population already at maximum size {self.population_size}"
            )
            
        self.population.append(genome)
        self.fitness_scores.append(0.0)
        
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get the history of population evolution.
        
        Returns:
            List of historical population statistics
        """
        return self.history.copy()


# Factory function for creating population managers
def create_population_manager(population_size: int = 100, max_generations: int = 1000) -> PopulationManager:
    """Create a new population manager instance.
    
    Args:
        population_size: Size of the population
        max_generations: Maximum number of generations
        
    Returns:
        Configured PopulationManager instance
    """
    return PopulationManager(population_size=population_size, max_generations=max_generations)


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = create_population_manager(population_size=10)
        
        # Define a simple genome factory
        def genome_factory():
            return DecisionGenome(
                parameters={'param1': 0.5, 'param2': 0.3},
                indicators=['SMA', 'EMA'],
                rules={'entry': 'SMA > EMA', 'exit': 'SMA < EMA'},
                risk_profile={'max_drawdown': 0.05}
            )
        
        # Initialize population
        manager.initialize_population(genome_factory)
        
        # Print initial statistics
        stats = manager.get_population_statistics()
        print("Initial population statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save state
        manager.save_state('population_state.json')
        
        # Load state
        manager.load_state('population_state.json')
        
        print("Population manager test completed successfully")
    
    asyncio.run(main())
