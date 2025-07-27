"""
Strategy Engine Interface Implementation

Provides the IStrategyEngine interface implementation for the EMP system.
This module serves as the main entry point for strategy engine functionality.

Author: EMP Development Team
Date: July 27, 2025
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

from .strategy_engine_impl import StrategyEngineImpl, create_strategy_engine

# Export the main implementation as StrategyEngine
StrategyEngine = StrategyEngineImpl

__all__ = ['StrategyEngine', 'create_strategy_engine']
