"""
EMP Sensory Cortex v2.2 - Main Package

The sensory cortex is the analytical brain of the EMP system, processing market data
through multiple dimensional engines to understand market behavior.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

# Core imports
from .core.base import (
    MarketData,
    DimensionalReading,
    MarketRegime,
    ConfidenceLevel,
    EventTier,
    SessionType,
    InstrumentMeta,
    EconomicEvent,
    OrderBookSnapshot,
    SystemHealth
)

# Import the new refactored engines
from .dimensions.how import HowEngine
from .dimensions.what import WhatEngine
from .dimensions.when import WhenEngine
from .dimensions.why import WhyEngine
from .dimensions.anomaly import AnomalyEngine

# Import legacy compatibility classes for backward compatibility
from .dimensions.compatibility import (
    InstitutionalMechanicsEngine,
    TechnicalRealityEngine,
    ChronalIntelligenceEngine,
    EnhancedFundamentalIntelligenceEngine,
    AnomalyIntelligenceEngine,
    MarketRegimeDetector,
    AdvancedPatternRecognition,
    TemporalAnalyzer,
    PatternRecognitionDetector,
    PatternType,
    AnomalyType
)

# Orchestration imports
try:
    from .orchestration.master_orchestrator import MasterOrchestrator
    from .orchestration.enhanced_intelligence_engine import ContextualFusionEngine
except ImportError:
    # Orchestration modules might not be fully implemented yet
    MasterOrchestrator = None
    ContextualFusionEngine = None

__version__ = "2.2.0"
__author__ = "EMP Development Team"

__all__ = [
    # Core classes
    'MarketData',
    'DimensionalReading', 
    'MarketRegime',
    'ConfidenceLevel',
    'EventTier',
    'SessionType',
    'InstrumentMeta',
    'EconomicEvent',
    'OrderBookSnapshot',
    'SystemHealth',
    
    # New refactored engines
    'HowEngine',
    'WhatEngine',
    'WhenEngine', 
    'WhyEngine',
    'AnomalyEngine',
    
    # Legacy compatibility classes
    'InstitutionalMechanicsEngine',
    'TechnicalRealityEngine',
    'ChronalIntelligenceEngine',
    'EnhancedFundamentalIntelligenceEngine',
    'AnomalyIntelligenceEngine',
    'MarketRegimeDetector',
    'AdvancedPatternRecognition',
    'TemporalAnalyzer',
    'PatternRecognitionDetector',
    'PatternType',
    'AnomalyType',
    
    # Orchestration (if available)
    'MasterOrchestrator',
    'ContextualFusionEngine'
]

