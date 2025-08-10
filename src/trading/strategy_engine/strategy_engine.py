"""
Canonical Strategy Engine entry point.
Use this module as the single strategy engine API surface.
"""

from src.core.strategy.engine import StrategyEngine as CoreStrategyEngine


def create_strategy_engine(*args, **kwargs) -> CoreStrategyEngine:
    return CoreStrategyEngine()


StrategyEngine = CoreStrategyEngine

__all__ = ['StrategyEngine', 'create_strategy_engine']
