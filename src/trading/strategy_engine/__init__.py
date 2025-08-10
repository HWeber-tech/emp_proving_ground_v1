"""
Canonical trading strategy engine facade that re-exports the core engine.
"""

from .strategy_engine import StrategyEngine

__all__ = [
    'StrategyEngine',
] 
