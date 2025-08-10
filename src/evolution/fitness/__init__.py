"""Deprecated fitness package pending consolidation into src/core/evolution."""
"""
Multi-dimensional fitness evaluation system for advanced evolution engine.

This package provides 7 specialized fitness dimensions for comprehensive
strategy evaluation and selection.
"""

from .base_fitness import IFitnessEvaluator, MockFitnessEvaluator
from .profit_fitness import ProfitFitness
from .survival_fitness import SurvivalFitness
from .adaptability_fitness import AdaptabilityFitness
from .robustness_fitness import RobustnessFitness
from .antifragility_fitness import AntifragilityFitness
from .efficiency_fitness import EfficiencyFitness
from .innovation_fitness import InnovationFitness
from .multi_dimensional_fitness_evaluator import MultiDimensionalFitnessEvaluator

__all__ = [
    'IFitnessEvaluator',
    'MockFitnessEvaluator',
    'BaseFitness',
    'FitnessResult',
    'ProfitFitness',
    'SurvivalFitness',
    'AdaptabilityFitness',
    'RobustnessFitness',
    'AntifragilityFitness',
    'EfficiencyFitness',
    'InnovationFitness',
    'MultiDimensionalFitnessEvaluator'
]
