"""
Sensory Cortex v2.2 - Production-Ready Market Intelligence System

A sophisticated market analysis system that understands markets through five interconnected dimensions:
- WHY: Fundamental forces driving market behavior
- HOW: Institutional mechanics and execution patterns  
- WHAT: Technical manifestations and price action
- WHEN: Temporal patterns and timing dynamics
- ANOMALY: Chaos, manipulation, and stress responses
"""

__version__ = "2.2.0"  # Updated for v2.2 production release
__author__ = "Market Intelligence Team"

# Export main classes for compatibility with existing code
from .orchestration.master_orchestrator import MasterOrchestrator as SensoryCortex
from .core.base import DimensionalReading as SensoryReading

# Export v2.2 dimensional engines
from .dimensions.why_engine import WHYEngine as EnhancedFundamentalIntelligenceEngine
from .dimensions.how_engine import HOWEngine as InstitutionalMechanicsEngine
from .dimensions.what_engine import WATEngine as TechnicalRealityEngine
from .dimensions.when_engine import WHENEngine as ChronalIntelligenceEngine
from .dimensions.anomaly_engine import ANOMALYEngine as AnomalyIntelligenceEngine

# Export core components
from .core.base import (
    MarketData, InstrumentMeta, OrderBookSnapshot, OrderBookLevel,
    MarketRegime, EconomicEvent, EventTier, EconomicData
)
from .core.utils import (
    EMA, WelfordVar, compute_confidence, normalize_signal,
    calculate_momentum, PerformanceTracker
)

# Also export the actual classes for new code
__all__ = [
    # Legacy compatibility
    'SensoryCortex',
    'SensoryReading', 
    'MasterOrchestrator',
    'DimensionalReading',
    
    # Enhanced dimensional engines
    'EnhancedFundamentalIntelligenceEngine',
    'InstitutionalMechanicsEngine',
    'TechnicalRealityEngine',
    'ChronalIntelligenceEngine',
    'AnomalyIntelligenceEngine',
    
    # Core components
    'MarketData',
    'InstrumentMeta',
    'OrderBookSnapshot',
    'OrderBookLevel',
    'MarketRegime',
    'EconomicEvent',
    'EventTier',
    'EconomicData',
    'EMA',
    'WelfordVar',
    'compute_confidence',
    'normalize_signal',
    'calculate_momentum',
    'PerformanceTracker'
]

