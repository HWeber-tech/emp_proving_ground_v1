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

# Also export the actual classes for new code
__all__ = [
    'SensoryCortex',
    'SensoryReading', 
    'ContextualFusionEngine',
    'DimensionalReading'
]

