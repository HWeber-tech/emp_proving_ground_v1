"""
EMP Proving Ground - Evolutionary Market Prediction System

A comprehensive trading system that combines:
- Risk Management Core
- PnL Engine
- 5D Sensory Cortex
- Evolutionary Decision Trees
- Adversarial Market Simulation
"""

__version__ = "2.0.0"
__author__ = "EMP Team"

# Core components
from .core import (
    RiskConfig,
    Instrument,
    InstrumentProvider,
    CurrencyConverter,
)

# Data pipeline
from .data import (
    TickDataStorage,
    TickDataCleaner,
    DukascopyIngestor,
)

# Evolution engine
from .evolution.real_genetic_engine import (
    TradingSignal,
    TradingStrategy,
    TechnicalIndicators,
    StrategyEvaluator,
    RealGeneticEngine,
)

# PnL engine
from .pnl import (
    TradeRecord,
    EnhancedPosition,
)

# Risk management
from .risk import (
    ValidationResult,
    RiskManager,
)

# Market simulation
from .simulation import (
    OrderSide,
    OrderType,
    OrderStatus,
    Order,
    MarketUnderstanding,
    MarketState,
    MarketSimulator,
    AdversarialEventType,
    AdversarialEvent,
    AdversarialEngine,
)

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
    # Evolution engine
    "RealGeneticEngine",
    "TradingSignal",
    "TradingStrategy",
    "TechnicalIndicators",
    "StrategyEvaluator",
    # Market simulation
    "MarketSimulator",
    "AdversarialEngine",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Order",
    "MarketUnderstanding",
    "MarketState",
    "AdversarialEventType",
    "AdversarialEvent",
    # Data pipeline
    "TickDataStorage",
    "TickDataCleaner",
    "DukascopyIngestor",
]
