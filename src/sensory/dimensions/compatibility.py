"""
Compatibility Layer - Legacy Engine Classes

This module provides compatibility classes that maintain backward compatibility
with the old monolithic sense file imports while using the new refactored structure.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.sensory.core.base import MarketData, DimensionalReading, MarketRegime, ConfidenceLevel
from .how.how_engine import HowEngine
from .what.what_engine import WhatEngine
from .when.when_engine import WhenEngine
from .why.why_engine import WhyEngine
from .anomaly.anomaly_engine import AnomalyEngine

logger = logging.getLogger(__name__)


# Legacy compatibility classes
class InstitutionalMechanicsEngine(HowEngine):
    """
    Legacy compatibility class for InstitutionalMechanicsEngine.
    Maps to the new HowEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("InstitutionalMechanicsEngine (legacy) initialized - using HowEngine")


class TechnicalRealityEngine(WhatEngine):
    """
    Legacy compatibility class for TechnicalRealityEngine.
    Maps to the new WhatEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("TechnicalRealityEngine (legacy) initialized - using WhatEngine")


class ChronalIntelligenceEngine(WhenEngine):
    """
    Legacy compatibility class for ChronalIntelligenceEngine.
    Maps to the new WhenEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("ChronalIntelligenceEngine (legacy) initialized - using WhenEngine")


class EnhancedFundamentalIntelligenceEngine(WhyEngine):
    """
    Legacy compatibility class for EnhancedFundamentalIntelligenceEngine.
    Maps to the new WhyEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("EnhancedFundamentalIntelligenceEngine (legacy) initialized - using WhyEngine")


class AnomalyIntelligenceEngine(AnomalyEngine):
    """
    Legacy compatibility class for AnomalyIntelligenceEngine.
    Maps to the new AnomalyEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("AnomalyIntelligenceEngine (legacy) initialized - using AnomalyEngine")


# Additional legacy classes that might be referenced
class MarketRegimeDetector(WhenEngine):
    """
    Legacy compatibility class for MarketRegimeDetector.
    Maps to the new WhenEngine (regime detection is part of when sense).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("MarketRegimeDetector (legacy) initialized - using WhenEngine")


class AdvancedPatternRecognition(AnomalyEngine):
    """
    Legacy compatibility class for AdvancedPatternRecognition.
    Maps to the new AnomalyEngine (pattern recognition is part of anomaly sense).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("AdvancedPatternRecognition (legacy) initialized - using AnomalyEngine")


class TemporalAnalyzer(WhenEngine):
    """
    Legacy compatibility class for TemporalAnalyzer.
    Maps to the new WhenEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("TemporalAnalyzer (legacy) initialized - using WhenEngine")


class PatternRecognitionDetector(AnomalyEngine):
    """
    Legacy compatibility class for PatternRecognitionDetector.
    Maps to the new AnomalyEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logger.info("PatternRecognitionDetector (legacy) initialized - using AnomalyEngine")


# Legacy enums and types that might be referenced
class MarketRegime:
    """Legacy MarketRegime enum - now using the one from core.base"""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    CONSOLIDATING = "consolidating"
    EXHAUSTED = "exhausted"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"
    VOLATILE = "volatile"
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    TRANSITION = "transition"
    CRISIS = "crisis"


class PatternType:
    """Legacy PatternType enum"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"
    CUP_AND_HANDLE = "cup_and_handle"
    UNKNOWN = "unknown"


class AnomalyType:
    """Legacy AnomalyType enum"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_SPIKE = "price_spike"
    GAP = "gap"
    DIVERGENCE = "divergence"
    OUTLIER = "outlier"
    UNKNOWN = "unknown"


# Export all legacy classes for backward compatibility
__all__ = [
    # Legacy engine classes
    'InstitutionalMechanicsEngine',
    'TechnicalRealityEngine', 
    'ChronalIntelligenceEngine',
    'EnhancedFundamentalIntelligenceEngine',
    'AnomalyIntelligenceEngine',
    'MarketRegimeDetector',
    'AdvancedPatternRecognition',
    'TemporalAnalyzer',
    'PatternRecognitionDetector',
    
    # Legacy enums
    'MarketRegime',
    'PatternType',
    'AnomalyType'
] 