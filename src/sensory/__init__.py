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

# Export core components
from .core.base import DimensionalReading as SensoryReading
from .core.base import (EconomicEvent, EventTier, InstrumentMeta, MarketData,
                        MarketRegime, OrderBookLevel, OrderBookSnapshot)
from .core.utils import (EMA, PerformanceTracker, WelfordVar,
                         calculate_momentum, compute_confidence,
                         normalize_signal)
from .dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine
from .dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
from .dimensions.enhanced_what_dimension import TechnicalRealityEngine
from .dimensions.enhanced_when_dimension import ChronalIntelligenceEngine
# Export v2.2 dimensional engines
from .dimensions.enhanced_why_dimension import \
    EnhancedFundamentalIntelligenceEngine
# Export main classes for compatibility with existing code
from .orchestration.master_orchestrator import \
    MasterOrchestrator as SensoryCortex

# Also export the actual classes for new code
__all__ = [
    # Legacy compatibility
    "SensoryCortex",
    "SensoryReading",
    "MasterOrchestrator",
    "DimensionalReading",
    # Enhanced dimensional engines
    "EnhancedFundamentalIntelligenceEngine",
    "InstitutionalMechanicsEngine",
    "TechnicalRealityEngine",
    "ChronalIntelligenceEngine",
    "AnomalyIntelligenceEngine",
    # Core components
    "MarketData",
    "InstrumentMeta",
    "OrderBookSnapshot",
    "OrderBookLevel",
    "MarketRegime",
    "EconomicEvent",
    "EventTier",
    "EMA",
    "WelfordVar",
    "compute_confidence",
    "normalize_signal",
    "calculate_momentum",
    "PerformanceTracker",
]
