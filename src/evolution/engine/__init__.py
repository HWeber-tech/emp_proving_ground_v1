"""
Evolution Engine Package
========================

This package contains the core evolution engine components for genetic algorithm
implementation in the EMP Proving Ground trading system.
"""

try:
    from src.core.population_manager import PopulationManager  # legacy
except Exception:  # pragma: no cover
    class PopulationManager:  # type: ignore
        pass
from .genetic_engine import GeneticEngine

__all__ = ['PopulationManager', 'GeneticEngine']
