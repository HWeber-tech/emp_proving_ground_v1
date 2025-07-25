#!/usr/bin/env python3
"""
GeneticEngine - Epic 2: Evolving "The Ambusher"
Evolution engine for creating hyper-specialized ambush strategies.
"""

import asyncio
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

from evolution.ambusher.ambusher_fitness import AmbusherFitnessFunction, AmbushEventType

logger = logging.getLogger(__name__)

@dataclass
class AmbusherGenome:
    """Genome structure for ambush strategies."""
    liquidity_grab_threshold: float = 0.001
    cascade_threshold: float = 0.002
    momentum_threshold: float = 0.0015
    volume_threshold: float = 2.0
    volume_spike: float = 3.0
    consecutive_moves: int = 3
    iceberg_threshold: float = 1000000
    risk_multiplier: float = 1.0
    position_size: float = 0.01
    stop_loss: float = 0.005
    take_profit: float = 0.01
    entry_delay: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary."""
        return {
            'liquidity_grab_threshold': self.liquidity_grab_threshold,
            'cascade_threshold': self.cascade_threshold,
            'momentum_threshold': self.momentum_threshold,
            'volume_threshold': self.volume_threshold,
            'volume_spike': self.volume_spike,
            'consecutive_moves': self.consecutive_moves,
            'iceberg_threshold': self.iceberg_threshold,
            'risk_multiplier': self.risk_multiplier,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_delay': self.entry_delay
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AmbusherGenome':
        """Create genome from dictionary."""
        return cls(**data)

class GeneticEngine:
    """Genetic algorithm engine for evolving ambush strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population_size = config.get('population_size', 100)
        self.generations = config.get('generations', 50)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.elite_size = config.get('elite_size', 10)
        
        self.fitness_function = AmbusherFitnessFunction(config.get('fitness', {}))
        self.population: List[Tuple[AmbusherGenome, float]] = []
        self.generation = 0
        
    def initialize_population(self) -> List[AmbusherGenome]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            genome = AmbusherGenome(
                liquidity_grab_threshold=random.uniform(0.0005, 0.003),
                cascade_threshold=random.uniform(0.001, 0.005),
                momentum_threshold=random.uniform(0.0008, 0.003),
                volume_threshold=random.uniform(1.5, 4.0),
                volume_spike=random.uniform(2.0, 5.0),
                consecutive_moves=random.randint(2, 5),
                iceberg_threshold=random.uniform(500000, 2000000),
                risk_multiplier=random.uniform(0.5, 2.0),
                position_size=random.uniform(0.005, 0.05),
                stop_loss=random.uniform(0.002, 0.01),
                take_profit=random.uniform(0.005, 0.02),
                entry_delay=random.randint(0, 60)
            )
            population.append(genome)
            
        return population
        
    def evaluate_population(self, market_data: Dict[str, Any], trade_history: List[Dict[str, Any]]) -> List[Tuple[AmbusherGenome, float]]:
        """Evaluate fitness of entire population."""
        evaluated = []
        
        for genome in [g for g, _ in self.population] if self.population else self.initialize_population():
            fitness = self.fitness_function.calculate_fitness(
                genome.to_dict(), market_data, trade_history
            )
            evaluated.append((genome, fitness))
            
        return sorted(evaluated, key=lambda x: x[1], reverse=True)
        
    def select_parents(self, evaluated_population: List[Tuple[AmbusherGenome, float]]) -> List[AmbusherGenome]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament = random.sample(evaluated_population, min(5, len(evaluated_population)))
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
            
        return parents
        
    def crossover(self, parent1: AmbusherGenome, parent2: AmbusherGenome) -> Tuple[AmbusherGenome, AmbusherGenome]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        # Create offspring
        offspring1 = AmbusherGenome()
        offspring2 = AmbusherGenome()
        
        # Crossover parameters
        params = ['liquidity_grab_threshold', 'cascade_threshold', 'momentum_threshold',
                 'volume_threshold', 'volume_spike', 'consecutive_moves', 'iceberg_threshold',
                 'risk_multiplier', 'position_size', 'stop_loss', 'take_profit', 'entry_delay']
        
        crossover_point = random.randint(1, len(params) - 1)
        
        for i, param in enumerate(params):
            if i < crossover_point:
                setattr(offspring1, param, getattr(parent1, param))
                setattr(offspring2, param, getattr(parent2, param))
            else:
                setattr(offspring1, param, getattr(parent2, param))
                setattr(offspring2, param, getattr(parent1, param))
                
        return offspring1, offspring2
        
    def mutate(self, genome: AmbusherGenome) -> AmbusherGenome:
        """Mutate a genome."""
        mutated = AmbusherGenome()
        
        # Copy all attributes
        for attr in ['liquidity_grab_threshold', 'cascade_threshold', 'momentum_threshold',
                    'volume_threshold', 'volume_spike', 'consecutive_moves', 'iceberg_threshold',
                    'risk_multiplier', 'position_size', 'stop_loss', 'take_profit', 'entry_delay']:
            value = getattr(genome, attr)
            
            if random.random() < self.mutation_rate:
                # Apply mutation
                if isinstance(value, float):
                    value *= random.uniform(0.8, 1.2)
                elif isinstance(value, int):
                    value += random.randint(-2, 2)
                    
            setattr(mutated, attr, value)
            
        return mutated
        
    def evolve(self, market_data: Dict[str, Any], trade_history: List[Dict[str, Any]]) -> Tuple[AmbusherGenome, Dict[str, Any]]:
        """Run the genetic algorithm to evolve the best strategy."""
        logger.info(f"Starting evolution with population size {self.population_size}")
        
        # Initialize population
        self.population = [(genome, 0.0) for genome in self.initialize_population()]
        
        best_genome = None
        best_fitness = 0.0
        
        for generation in range(self.generations):
            # Evaluate population
            self.population = self.evaluate_population(market_data, trade_history)
            
            # Track best
            current_best = self.population[0]
            if current_best[1] > best_fitness:
                best_genome = current_best[0]
                best_fitness = current_best[1]
                
            logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.4f}")
            
            # Select parents
            parents = self.select_parents(self.population)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best performers
            for elite in self.population[:self.elite_size]:
                new_population.append(elite)
                
            # Generate offspring
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.extend([(offspring1, 0.0), (offspring2, 0.0)])
                
            # Ensure population size
            self.population = new_population[:self.population_size]
            
        # Final evaluation
        self.population = self.evaluate_population(market_data, trade_history)
        best_genome = self.population[0][0]
        best_fitness = self.population[0][1]
        
        # Get evolution summary
        summary = {
            'best_genome': best_genome.to_dict(),
            'best_fitness': best_fitness,
            'generations': self.generations,
            'population_size': self.population_size,
            'final_population': [(g.to_dict(), f) for g, f in self.population[:10]]
        }
        
        logger.info(f"Evolution complete. Best fitness: {best_fitness:.4f}")
        return best_genome, summary
        
    def save_genome(self, genome: AmbusherGenome, path: str):
        """Save genome to file."""
        with open(path, 'w') as f:
            json.dump(genome.to_dict(), f, indent=2)
            
    def load_genome(self, path: str) -> AmbusherGenome:
        """Load genome from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return AmbusherGenome.from_dict(data)
