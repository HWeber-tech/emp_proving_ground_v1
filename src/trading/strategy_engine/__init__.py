"""
Advanced Trading Strategy Engine Package

This package implements the complete strategy engine architecture as specified
in Phase 3 with proper modular decomposition.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

# Import strategy templates
from .templates.trend_following import TrendFollowingStrategy
from .templates.mean_reversion import MeanReversionStrategy
from .templates.momentum import MomentumStrategy

# Import optimization modules
from .optimization.genetic_optimizer import GeneticOptimizer
from .optimization.parameter_tuning import ParameterTuner

# Import backtesting modules
from .backtesting.backtest_engine import BacktestEngine
from .backtesting.performance_analyzer import PerformanceAnalyzer

# Import live management modules
from .live_management.strategy_monitor import StrategyMonitor
from .live_management.dynamic_adjustment import DynamicAdjustment

# Import main engine
from .strategy_engine import StrategyEngine

__all__ = [
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
] 