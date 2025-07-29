"""
EMP Selection Strategies v1.1

Modular selection strategies for the adaptive core.
Separates selection logic from the main evolution engine.
"""

import numpy as np
import random
from typing import List, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from ...genome.models.genome import StrategyGenome
from ...core.events import FitnessReport

logger = logging.getLogger(__name__)


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.selection_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def select(self, population: List[StrategyGenome], 
              fitness_reports: List[FitnessReport], 
              selection_size: int) -> List[StrategyGenome]:
        """Select individuals from population based on fitness."""
        pass
        
    def log_selection(self, selected: List[StrategyGenome], 
                     population_size: int, generation: int):
        """Log selection event."""
        entry = {
            "timestamp": datetime.now(),
            "strategy": self.name,
            "selected_count": len(selected),
            "population_size": population_size,
            "generation": generation,
            "selected_ids": [genome.id for genome in selected]
        }
        self.selection_history.append(entry)
        logger.debug(f"Selection logged: {self.name} selected {len(selected)} from {population_size}")


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy."""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__("tournament_selection")
        self.tournament_size = tournament_size
        
    def select(self, population: List[StrategyGenome], 
              fitness_reports: List[FitnessReport], 
              selection_size: int) -> List[StrategyGenome]:
        """Select using tournament selection."""
        if len(population) < self.tournament_size:
            logger.warning(f"Population size {len(population)} smaller than tournament size {self.tournament_size}")
            return population[:selection_size]
            
        # Create fitness lookup
        fitness_lookup = {report.genome_id: report.fitness_score for report in fitness_reports}
        
        selected = []
        for _ in range(selection_size):
            # Select tournament participants
            tournament = random.sample(population, self.tournament_size)
            
            # Find winner (highest fitness)
            winner = max(tournament, key=lambda g: fitness_lookup.get(g.id, 0))
            selected.append(winner)
            
        return selected


class RouletteWheelSelection(SelectionStrategy):
    """Roulette wheel (fitness proportional) selection."""
    
    def __init__(self):
        super().__init__("roulette_wheel_selection")
        
    def select(self, population: List[StrategyGenome], 
              fitness_reports: List[FitnessReport], 
              selection_size: int) -> List[StrategyGenome]:
        """Select using roulette wheel selection."""
        if not fitness_reports:
            logger.warning("No fitness reports for roulette wheel selection")
            return population[:selection_size]
            
        # Create fitness lookup
        fitness_lookup = {report.genome_id: report.fitness_score for report in fitness_reports}
        
        # Calculate selection probabilities
        fitness_values = [fitness_lookup.get(genome.id, 0) for genome in population]
        total_fitness = sum(fitness_values)
        
        if total_fitness <= 0:
            logger.warning("Zero or negative total fitness, using uniform selection")
            return random.sample(population, min(selection_size, len(population)))
            
        probabilities = [f / total_fitness for f in fitness_values]
        
        # Select individuals
        selected = []
        for _ in range(selection_size):
            chosen = np.random.choice(population, p=probabilities)
            selected.append(chosen)
            
        return selected


class RankBasedSelection(SelectionStrategy):
    """Rank-based selection strategy."""
    
    def __init__(self, selection_pressure: float = 1.5):
        super().__init__("rank_based_selection")
        self.selection_pressure = selection_pressure
        
    def select(self, population: List[StrategyGenome], 
              fitness_reports: List[FitnessReport], 
              selection_size: int) -> List[StrategyGenome]:
        """Select using rank-based selection."""
        if not fitness_reports:
            logger.warning("No fitness reports for rank-based selection")
            return population[:selection_size]
            
        # Create fitness lookup
        fitness_lookup = {report.genome_id: report.fitness_score for report in fitness_reports}
        
        # Sort population by fitness
        sorted_population = sorted(population, key=lambda g: fitness_lookup.get(g.id, 0), reverse=True)
        
        # Calculate rank probabilities
        n = len(sorted_population)
        probabilities = []
        
        for i in range(n):
            rank = i + 1
            prob = (2 - self.selection_pressure) / n + (2 * rank * (self.selection_pressure - 1)) / (n * (n - 1))
            probabilities.append(prob)
            
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / n] * n
            
        # Select individuals
        selected = []
        for _ in range(selection_size):
            chosen = np.random.choice(sorted_population, p=probabilities)
            selected.append(chosen)
            
        return selected


class ElitistSelection(SelectionStrategy):
    """Elitist selection strategy."""
    
    def __init__(self, elite_size: int = 1):
        super().__init__("elitist_selection")
        self.elite_size = elite_size
        
    def select(self, population: List[StrategyGenome], 
              fitness_reports: List[FitnessReport], 
              selection_size: int) -> List[StrategyGenome]:
        """Select using elitist selection."""
        if not fitness_reports:
            logger.warning("No fitness reports for elitist selection")
            return population[:selection_size]
            
        # Create fitness lookup
        fitness_lookup = {report.genome_id: report.fitness_score for report in fitness_reports}
        
        # Sort population by fitness
        sorted_population = sorted(population, key=lambda g: fitness_lookup.get(g.id, 0), reverse=True)
        
        # Select elite individuals
        elite_count = min(self.elite_size, selection_size)
        selected = sorted_population[:elite_count]
        
        # Fill remaining slots with random selection
        remaining_slots = selection_size - elite_count
        if remaining_slots > 0 and len(sorted_population) > elite_count:
            remaining = sorted_population[elite_count:]
            selected.extend(random.sample(remaining, min(remaining_slots, len(remaining))))
            
        return selected


class SelectionFactory:
    """Factory for creating selection strategies."""
    
    _strategies = {
        "tournament": TournamentSelection,
        "roulette_wheel": RouletteWheelSelection,
        "rank_based": RankBasedSelection,
        "elitist": ElitistSelection
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, **kwargs) -> SelectionStrategy:
        """Create a selection strategy by name."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown selection strategy: {strategy_name}")
            
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)
        
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available selection strategies."""
        return list(cls._strategies.keys())
        
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a new selection strategy."""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered selection strategy: {name}") 
