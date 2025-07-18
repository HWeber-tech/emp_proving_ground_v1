"""
Evolution package for EMP system.

This package contains:
- Real genetic programming engine
- Strategy evolution and optimization
- Trading strategy evaluation
"""

from .real_genetic_engine import (RealGeneticEngine, StrategyEvaluator,
                                  TechnicalIndicators, TradingStrategy)

# Backward compatibility aliases
EvolutionEngine = RealGeneticEngine
DecisionGenome = TradingStrategy
FitnessEvaluator = StrategyEvaluator


class EvolutionConfig:
    """Configuration for evolution engine"""

    def __init__(
        self,
        population_size=100,
        elite_ratio=0.1,
        crossover_ratio=0.6,
        mutation_ratio=0.3,
    ):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio


__all__ = [
    "RealGeneticEngine",
    "TradingStrategy",
    "StrategyEvaluator",
    "TechnicalIndicators",
    "EvolutionEngine",
    "DecisionGenome",
    "EvolutionConfig",
    "FitnessEvaluator",
]
