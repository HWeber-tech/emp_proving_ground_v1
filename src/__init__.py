"""
EMP Proving Ground - Evolutionary Market Prediction System

A comprehensive trading system that combines:
- Risk Management Core
- PnL Engine  
- 4D+1 Sensory Cortex
- Evolutionary Decision Trees
- Adversarial Market Simulation
"""

__version__ = "2.0.0"
__author__ = "EMP Team"

from .core import *
from .risk import *
from .pnl import *
from .sensory import *
from .evolution import *
from .simulation import *
from .data import *

__all__ = [
    # Core components
    "RiskConfig",
    "Instrument", 
    "InstrumentProvider",
    "CurrencyConverter",
    
    # Risk management
    "RiskManager",
    "ValidationResult",
    
    # PnL engine
    "EnhancedPosition",
    "TradeRecord",
    
    # Sensory cortex
    "SensoryCortex",
    "SensoryReading",
    
    # Evolution engine
    "EvolutionEngine",
    "DecisionGenome",
    "FitnessEvaluator",
    
    # Market simulation
    "MarketSimulator",
    "AdversarialEngine",
    
    # Data pipeline
    "TickDataStorage",
    "TickDataCleaner",
    "DukascopyIngestor",
] 