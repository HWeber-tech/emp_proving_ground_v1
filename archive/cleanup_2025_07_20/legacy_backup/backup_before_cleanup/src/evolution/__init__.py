"""
Evolution package for EMP system.

This package contains:
- Real genetic programming engine
- Strategy evolution and optimization
- Trading strategy evaluation
"""

from .real_genetic_engine import RealGeneticEngine, TradingStrategy, StrategyEvaluator, TechnicalIndicators

__all__ = ['RealGeneticEngine', 'TradingStrategy', 'StrategyEvaluator', 'TechnicalIndicators'] 