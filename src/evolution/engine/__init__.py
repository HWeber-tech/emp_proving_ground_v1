"""
Evolution Engine Package (legacy compatibility)

Re-exports consolidated core evolution components to maintain import
compatibility while centralizing implementation under `src.core.evolution`.
"""

from src.core.evolution.engine import EvolutionConfig, EvolutionEngine  # type: ignore
from src.core.evolution.fitness import *  # noqa: F401,F403
from src.core.evolution.operators import *  # noqa: F401,F403
from src.core.evolution.population import Population  # type: ignore

__all__ = [
    'EvolutionEngine',
    'EvolutionConfig',
    'Population',
]
