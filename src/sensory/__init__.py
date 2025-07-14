"""
Multidimensional Market Intelligence System

A sophisticated market analysis system that understands markets through five interconnected dimensions:
- WHY: Fundamental forces driving market behavior
- HOW: Institutional mechanics and execution patterns  
- WHAT: Technical manifestations and price action
- WHEN: Temporal patterns and timing dynamics
- ANOMALY: Chaos, manipulation, and stress responses
"""

__version__ = "1.0.0"
__author__ = "Market Intelligence Team"

# Export main classes for compatibility with existing code
from .orchestration.enhanced_intelligence_engine import ContextualFusionEngine as SensoryCortex
from .core.base import DimensionalReading as SensoryReading

# Export enhanced dimensional engines
from .dimensions.enhanced_why_dimension import EnhancedFundamentalIntelligenceEngine
from .dimensions.enhanced_how_dimension import InstitutionalMechanicsEngine
from .dimensions.enhanced_what_dimension import TechnicalRealityEngine
from .dimensions.enhanced_when_dimension import ChronalIntelligenceEngine
from .dimensions.enhanced_anomaly_dimension import AnomalyIntelligenceEngine

# Also export the actual classes for new code
__all__ = [
    'SensoryCortex',
    'SensoryReading', 
    'ContextualFusionEngine',
    'DimensionalReading',
    'EnhancedFundamentalIntelligenceEngine',
    'InstitutionalMechanicsEngine',
    'TechnicalRealityEngine',
    'ChronalIntelligenceEngine',
    'AnomalyIntelligenceEngine'
]

