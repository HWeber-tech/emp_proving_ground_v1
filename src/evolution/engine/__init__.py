"""
Evolution Engine Package (legacy compatibility)

Re-exports consolidated core evolution components to maintain import
compatibility while centralizing implementation under `src.core.evolution`.
"""

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.evolution.fitness import *  # noqa: F401,F403
from src.core.evolution.operators import *  # noqa: F401,F403
from src.core.evolution.population import Population

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "Population",
]
