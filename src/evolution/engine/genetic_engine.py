"""
EMP Genetic Engine v1.1

Modular genetic engine for the adaptive core.
Orchestrates selection, variation, and population management components.
"""

import asyncio
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ...genome.models.genome import DecisionGenome
from ...core.events import FitnessReport, EvolutionEvent
from ...core.event_bus import publish_event, EventType
from ...simulation.evaluation.fitness_evaluator import FitnessEvaluator, EvaluationContext
from .population_manager import PopulationManager
from ..selection.selection_strategies import SelectionStrategy, SelectionFactory
from ..variation.variation_strategies import CrossoverStrategy, MutationStrategy, VariationFactory

logger = logging.getLogger(__name__)


class GeneticEngine:
    """Modular genetic engine orchestrating evolution components."""
    
    def __init__(self, 
                 population_size: int = 100,
                 elite_size: int = 10,
                 selection_strategy: str = "tournament",
                 crossover_strategy: str = "uniform",
                 mutation_strategy: str = "gaussian",
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 **kwargs):
        
        # Initialize population manager
        self.population_manager = PopulationManager(population_size, elite_size)
        
        # Initialize selection strategy
        self.selection_strategy = SelectionFactory.create_strategy(
            selection_strategy, **kwargs.get('selection_kwargs', {})
        )
        
        # Initialize variation strategies
        self.crossover_strategy = VariationFactory.create_crossover_strategy(
            crossover_strategy, **kwargs.get('crossover_kwargs', {})
        )
        self.mutation_strategy = VariationFactory.create_mutation_strategy(
            mutation_strategy, **kwargs.get('mutation_kwargs', {})
        )
        
        # Initialize fitness evaluator (simulation envelope)
        self.fitness_evaluator = FitnessEvaluator()
        
        # Evolution parameters
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        # Evolution history
        self.evolution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Genetic Engine initialized with {population_size} population size")
        
    def initialize_population(self, genome_factory: Callable[[], DecisionGenome]):
        """Initialize population with random genomes."""
        self.population_manager.initialize_population(genome_factory)
        self.generation = 0
        
    async def evolve_generation(self, market_data: Optional[Dict[str, Any]] = None) -> List[DecisionGenome]:
        """Evolve one generation using the simulation envelope for fitness evaluation."""
        try:
            # Step 1: Evaluate current population using simulation envelope
            fitness_reports = await self._evaluate_population(market_data)
            
            # Step 2: Update fitness cache in population manager
            self.population_manager.update_fitness_cache(fitness_reports)
            
            # Step 3: Get current population statistics
            stats = self.population_manager.get_population_statistics()
            
            # Step 4: Create new population through selection and variation
            new_population = await self._create_new_population()
            
            # Step 5: Update population
            self.population_manager.population = new_population
            
            # Step 6: Advance generation
            self.population_manager.advance_generation()
            self.generation += 1
            
            # Step 7: Log evolution
            self._log_evolution(stats)
            
            # Step 8: Publish evolution event
            await self._publish_evolution_event(stats)
            
            logger.info(f"Evolved generation {self.generation} with {len(new_population)} genomes")
            return new_population
            
        except Exception as e:
            logger.error(f"Error evolving generation: {e}")
            return self.population_manager.population
            
    async def _evaluate_population(self, market_data: Optional[Dict[str, Any]] = None) -> List[FitnessReport]:
        """Evaluate population using simulation envelope."""
        fitness_reports = []
        population = self.population_manager.get_population()
        
        for genome in population:
            try:
                # Create evaluation context
                context = EvaluationContext(
                    strategy_id=f"strategy_{genome.genome_id}",
                    genome_id=genome.genome_id,
                    generation=self.generation,
                    initial_capital=100000.0,  # Default initial capital
                    evaluation_period=252,  # Default evaluation period
                    market_data=market_data
                )
                
                # Evaluate genome using simulation envelope
                if market_data and "backtest_results" in market_data:
                    # Use backtest results if available
                    fitness_report = await self.fitness_evaluator.evaluate_backtest_fitness(
                        market_data["backtest_results"], context
                    )
                else:
                    # Create empty trade history for basic evaluation
                    empty_trade_history = []
                    fitness_report = await self.fitness_evaluator.evaluate_fitness(
                        empty_trade_history, context
                    )
                    
                fitness_reports.append(fitness_report)
                
            except Exception as e:
                logger.error(f"Error evaluating genome {genome.genome_id}: {e}")
                # Create default fitness report
                context = EvaluationContext(
                    strategy_id=f"strategy_{genome.genome_id}",
                    genome_id=genome.genome_id,
                    generation=self.generation,
                    initial_capital=100000.0,
                    evaluation_period=252
                )
                default_report = self.fitness_evaluator._create_default_fitness_report(context)
                fitness_reports.append(default_report)
                
        return fitness_reports
        
    async def _create_new_population(self) -> List[DecisionGenome]:
        """Create new population through selection, crossover, and mutation."""
        new_population = []
        population_size = self.population_manager.population_size
        elite_size = self.population_manager.elite_size
        
        # Elitism: preserve best individuals
        elite = self.population_manager.get_best_genomes(elite_size)
        new_population.extend(elite)
        
        # Generate remaining individuals
        remaining_size = population_size - elite_size
        
        while len(new_population) < population_size:
            # Selection
            parents = self.selection_strategy.select(
                self.population_manager.population,
                [],  # Fitness reports not needed here as we use cache
                selection_size=2
            )
            
            if len(parents) >= 2:
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover_strategy.crossover(parents[0], parents[1])
                    
                    # Mutation
                    child1 = self.mutation_strategy.mutate(child1, self.mutation_rate)
                    child2 = self.mutation_strategy.mutate(child2, self.mutation_rate)
                    
                    new_population.extend([child1, child2])
                else:
                    # No crossover, just mutation
                    child1 = self.mutation_strategy.mutate(parents[0], self.mutation_rate)
                    child2 = self.mutation_strategy.mutate(parents[1], self.mutation_rate)
                    
                    new_population.extend([child1, child2])
            else:
                # Fallback: create random individuals
                logger.warning("Insufficient parents for crossover, creating random individuals")
                break
                
        # Ensure population size
        while len(new_population) < population_size:
            # Create random individual (simplified)
            random_genome = DecisionGenome(
                genome_id=f"random_gen{self.generation}_id{len(new_population)}"
            )
            new_population.append(random_genome)
            
        return new_population[:population_size]
        
    def _log_evolution(self, stats: Dict[str, Any]):
        """Log evolution statistics."""
        entry = {
            'generation': self.generation,
            'timestamp': datetime.now(),
            'statistics': stats,
            'selection_strategy': self.selection_strategy.name,
            'crossover_strategy': self.crossover_strategy.name,
            'mutation_strategy': self.mutation_strategy.name,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate
        }
        
        self.evolution_history.append(entry)
        
    async def _publish_evolution_event(self, stats: Dict[str, Any]):
        """Publish evolution event."""
        event = EvolutionEvent(
            timestamp=datetime.now(),
            event_type="generation_complete",
            genome_id="",  # Will be set by caller
            generation=self.generation,
            population_size=stats['population_size'],
            best_fitness=stats['best_fitness'],
            average_fitness=stats['average_fitness'],
            metadata={
                'selection_strategy': self.selection_strategy.name,
                'crossover_strategy': self.crossover_strategy.name,
                'mutation_strategy': self.mutation_strategy.name,
                'diversity': stats['diversity']
            }
        )
        
        # Note: In a real implementation, this would be properly async
        # await publish_event(event)
        
    def get_best_genome(self) -> Optional[DecisionGenome]:
        """Get the best genome from current population."""
        best_genomes = self.population_manager.get_best_genomes(1)
        return best_genomes[0] if best_genomes else None
        
    def get_population(self) -> List[DecisionGenome]:
        """Get current population."""
        return self.population_manager.population.copy()
        
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get current population statistics."""
        return self.population_manager.get_population_statistics()
        
    def get_evolution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get evolution history."""
        if limit:
            return self.evolution_history[-limit:]
        return self.evolution_history.copy()
        
    def change_selection_strategy(self, strategy_name: str, **kwargs):
        """Change selection strategy."""
        self.selection_strategy = SelectionFactory.create_strategy(strategy_name, **kwargs)
        logger.info(f"Changed selection strategy to {strategy_name}")
        
    def change_crossover_strategy(self, strategy_name: str, **kwargs):
        """Change crossover strategy."""
        self.crossover_strategy = VariationFactory.create_crossover_strategy(strategy_name, **kwargs)
        logger.info(f"Changed crossover strategy to {strategy_name}")
        
    def change_mutation_strategy(self, strategy_name: str, **kwargs):
        """Change mutation strategy."""
        self.mutation_strategy = VariationFactory.create_mutation_strategy(strategy_name, **kwargs)
        logger.info(f"Changed mutation strategy to {strategy_name}")
        
    def update_parameters(self, crossover_rate: Optional[float] = None, 
                         mutation_rate: Optional[float] = None):
        """Update evolution parameters."""
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
            
        logger.info(f"Updated parameters: crossover_rate={self.crossover_rate}, mutation_rate={self.mutation_rate}")
        
    def reset_evolution(self):
        """Reset evolution state."""
        self.population_manager.reset_population()
        self.generation = 0
        self.evolution_history = []
        logger.info("Evolution reset")
        
    def export_state(self) -> Dict[str, Any]:
        """Export current evolution state."""
        return {
            'population': self.population_manager.export_population(),
            'evolution_history': self.evolution_history,
            'generation': self.generation,
            'parameters': {
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'selection_strategy': self.selection_strategy.name,
                'crossover_strategy': self.crossover_strategy.name,
                'mutation_strategy': self.mutation_strategy.name
            }
        }
        
    def import_state(self, data: Dict[str, Any]):
        """Import evolution state."""
        self.population_manager.import_population(data.get('population', {}))
        self.evolution_history = data.get('evolution_history', [])
        self.generation = data.get('generation', 0)
        
        params = data.get('parameters', {})
        self.crossover_rate = params.get('crossover_rate', self.crossover_rate)
        self.mutation_rate = params.get('mutation_rate', self.mutation_rate)
        
        logger.info("Evolution state imported") 