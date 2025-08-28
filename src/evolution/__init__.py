"""
EMP Evolution Layer v1.1

The Evolution Layer orchestrates the genetic programming and evolution of
trading strategies. It manages populations, selection, variation, and
evaluation to drive continuous improvement and adaptation.

Architecture:
- engine/: Genetic engine and population management
- selection/: Selection algorithms (tournament, fitness proportionate)
- variation/: Crossover, mutation, and recombination operators
- evaluation/: Fitness evaluation and backtesting
- meta/: Meta-evolution for self-improving evolution
"""

# Legacy facade: re-export core evolution interfaces
_ = None  # Legacy facade: re-export core evolution interfaces
from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.fitness import FitnessEvaluator
from src.core.evolution.operators import *  # noqa: F401,F403
from src.core.evolution.population import Population

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Evolution Layer - Genetic Programming and Evolution (core-consolidated)" 

__all__ = [
    'EvolutionEngine',
    'EvolutionConfig',
    'FitnessEvaluator',
    'Population',
]
