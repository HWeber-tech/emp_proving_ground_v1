"""
Canonical trading strategy engine facade that re-exports the core engine.
"""

from .strategy_engine import StrategyEngine

__all__ = [
<<<<<<< Current (Your changes)
    # Strategy templates
    'TrendFollowingStrategy',
    'MeanReversionStrategy', 
    'MomentumStrategy',
    
    # Optimization
    'GeneticOptimizer',
    'ParameterTuner',
    
    # Backtesting
    'BacktestEngine',
    'PerformanceAnalyzer',
    
    # Live management
    'StrategyMonitor',
    'DynamicAdjustment',
    
    # Main engine
    'StrategyEngine'
=======
    'StrategyEngine',
>>>>>>> Incoming (Background Agent changes)
] 
