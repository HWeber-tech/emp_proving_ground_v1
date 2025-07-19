"""
EMP Tournament Selection v1.1

Tournament selection strategy implementation for the evolutionary process.
Implements the ISelectionStrategy interface for selecting genomes based on fitness.
"""

import logging
import random
from typing import List, Optional
from dataclasses import dataclass

from src.core.interfaces import ISelectionStrategy
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


@dataclass
class TournamentSelectionConfig:
    """Configuration for tournament selection."""
    tournament_size: int = 3
    selection_pressure: float = 1.0  # 1.0 = standard, >1.0 = more selective
    allow_duplicates: bool = False


class TournamentSelection(ISelectionStrategy):
    """
    Tournament selection strategy for genetic algorithms.
    
    Implements tournament selection where k individuals are randomly selected
    from the population and the fittest one is chosen as the winner.
    """
    
    def __init__(self, tournament_size: int = 3, selection_pressure: float = 1.0):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
            selection_pressure: How strongly to favor better individuals (1.0 = standard)
        """
        self.tournament_size = max(2, tournament_size)
        self.selection_pressure = max(0.1, selection_pressure)
        
        logger.info(f"TournamentSelection initialized with size {tournament_size}")
    
    def select(self, population: List[DecisionGenome], fitness_scores: List[float]) -> DecisionGenome:
        """
        Select a genome using tournament selection.
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            
        Returns:
            Selected genome
            
        Raises:
            ValueError: If population is empty or sizes don't match
        """
        if not population:
            raise ValueError("Cannot select from empty population")
        
        if len(population) != len(fitness_scores):
            raise ValueError("Population and fitness scores must have same length")
        
        # Ensure tournament size doesn't exceed population
        actual_tournament_size = min(self.tournament_size, len(population))
        
        # Select tournament participants
        tournament_indices = random.sample(range(len(population)), actual_tournament_size)
        tournament_genomes = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Find the winner (highest fitness)
        winner_index = max(range(len(tournament_fitness)), key=lambda i: tournament_fitness[i])
        winner = tournament_genomes[winner_index]
        
        logger.debug(f"Tournament selection: selected genome with fitness {tournament_fitness[winner_index]}")
        
        return winner
    
    def select_multiple(self, population: List[DecisionGenome], fitness_scores: List[float], count: int) -> List[DecisionGenome]:
        """
        Select multiple genomes using tournament selection.
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            count: Number of genomes to select
            
        Returns:
            List of selected genomes
        """
        selected = []
        for _ in range(count):
            winner = self.select(population, fitness_scores)
            selected.append(winner)
        
        return selected
    
    def select_with_replacement(self, population: List[DecisionGenome], fitness_scores: List[float], count: int) -> List[DecisionGenome]:
        """
        Select genomes with replacement (allowing duplicates).
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            count: Number of genomes to select
            
        Returns:
            List of selected genomes
        """
        selected = []
        for _ in range(count):
            winner = self.select(population, fitness_scores)
            selected.append(winner)
        
        return selected
    
    def select_without_replacement(self, population: List[DecisionGenome], fitness_scores: List[float], count: int) -> List[DecisionGenome]:
        """
        Select genomes without replacement (no duplicates).
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            count: Number of genomes to select
            
        Returns:
            List of selected genomes
            
        Raises:
            ValueError: If count exceeds population size
        """
        if count > len(population):
            raise ValueError("Cannot select more genomes than population size")
        
        # Create working copies
        working_population = population.copy()
        working_fitness = fitness_scores.copy()
        
        selected = []
        for _ in range(count):
            winner = self.select(working_population, working_fitness)
            selected.append(winner)
            
            # Remove winner from working population
            winner_index = working_population.index(winner)
            working_population.pop(winner_index)
            working_fitness.pop(winner_index)
        
        return selected
    
    def select_proportional(self, population: List[DecisionGenome], fitness_scores: List[float], count: int) -> List[DecisionGenome]:
        """
        Select genomes proportionally to their fitness (roulette wheel).
        
        Args:
            population: List of genomes to select from
            fitness_scores: Corresponding fitness scores for each genome
            count: Number of genomes to select
            
        Returns:
            List of selected genomes
        """
        if not population:
            raise ValueError("Cannot select from empty population")
        
        # Ensure all fitness scores are non-negative
        min_fitness = min(fitness_scores)
        adjusted_scores = [f - min_fitness + 1e-10 for f in fitness_scores]  # Add small epsilon
        
        total_fitness = sum(adjusted_scores)
        if total_fitness <= 0:
            # If all fitness scores are equal, use uniform selection
            return random.choices(population, k=count)
        
        # Select proportionally
        selected = random.choices(
            population,
            weights=adjusted_scores,
            k=count
        )
        
        return selected
    
    @property
    def name(self) -> str:
        """Return the name of this selection strategy."""
        return f"TournamentSelection(tournament_size={self.tournament_size})"
    
    def __repr__(self) -> str:
        """String representation of the selection strategy."""
        return f"TournamentSelection(tournament_size={self.tournament_size}, selection_pressure={self.selection_pressure})"
