"""
EMP Proving Ground - World-Class Trading System

A production-ready algorithmic trading system featuring:
- Real FIX API Integration
- Genetic Evolution Engine
- Multi-Dimensional Fitness Evaluation
- Enterprise Exception Handling
- Comprehensive Validation Framework
"""

__version__ = "2.0.0"
__author__ = "EMP Team"

from .core import *
from .trading import *
from .evolution import *
from .operational import *

__all__ = [
    # Core components
    "PopulationManager",
    "DecisionGenome",
    "EMPException",
    "TradingException",
    "ValidationManager",
    
    # Trading components
    "OrderExecutionEngine",
    "RiskManager",
    "PositionSizer",
    
    # Evolution engine
    "FitnessEvaluator",
    "GeneticOperator",
    
    # Operational components
    "ICMarketsAPI",
    "ICMarketsConfig",
] 