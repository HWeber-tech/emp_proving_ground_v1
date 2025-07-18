"""
Compatibility Layer - Backward Compatibility

This module provides backward compatibility for old import paths
by mapping legacy class names to the new refactored engine classes.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Backward Compatibility Fix
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.sensory.core.base import MarketData, DimensionalReading, MarketRegime, ConfidenceLevel

logger = logging.getLogger(__name__)


class InstitutionalMechanicsEngine:
    """
    Legacy compatibility class for InstitutionalMechanicsEngine.
    Maps to the new HowEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        try:
            from .how.how_engine import HowEngine
            self._engine = HowEngine(config)
            logger.info("InstitutionalMechanicsEngine (legacy) initialized - using HowEngine")
        except ImportError as e:
            logger.error(f"Failed to import HowEngine: {e}")
            self._engine = None
    
    def analyze_market_data(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze market data using the new HowEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_market_data(market_data, symbol)
    
    def analyze_institutional_mechanics(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze institutional mechanics using the new HowEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_institutional_mechanics(market_data, symbol)
    
    def get_dimensional_reading(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> DimensionalReading:
        """Get dimensional reading using the new HowEngine"""
        if self._engine is None:
            return DimensionalReading(
                dimension="HOW",
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.0,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Engine not available"]
            )
        return self._engine.get_dimensional_reading(market_data, symbol)
    
    def get_dimension_name(self) -> str:
        """Get dimension name"""
        return "HOW"
    
    def reset(self) -> None:
        """Reset the engine"""
        if self._engine is not None:
            # Reset logic if needed
            pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Get engine snapshot"""
        return {
            'dimension': 'HOW',
            'engine_type': 'InstitutionalMechanicsEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def update(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Update with new market data"""
        return self.analyze_market_data(market_data, symbol)


class TechnicalRealityEngine:
    """
    Legacy compatibility class for TechnicalRealityEngine.
    Maps to the new WhatEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        try:
            from .what.what_engine import WhatEngine
            self._engine = WhatEngine(config)
            logger.info("TechnicalRealityEngine (legacy) initialized - using WhatEngine")
        except ImportError as e:
            logger.error(f"Failed to import WhatEngine: {e}")
            self._engine = None
    
    def analyze_market_data(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze market data using the new WhatEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_market_data(market_data, symbol)
    
    def analyze_technical_reality(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze technical reality using the new WhatEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_technical_reality(market_data, symbol)
    
    def get_dimensional_reading(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> DimensionalReading:
        """Get dimensional reading using the new WhatEngine"""
        if self._engine is None:
            return DimensionalReading(
                dimension="WHAT",
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.0,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Engine not available"]
            )
        return self._engine.get_dimensional_reading(market_data, symbol)
    
    def get_dimension_name(self) -> str:
        """Get dimension name"""
        return "WHAT"
    
    def reset(self) -> None:
        """Reset the engine"""
        if self._engine is not None:
            # Reset logic if needed
            pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Get engine snapshot"""
        return {
            'dimension': 'WHAT',
            'engine_type': 'TechnicalRealityEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def update(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Update with new market data"""
        return self.analyze_market_data(market_data, symbol)


class ChronalIntelligenceEngine:
    """
    Legacy compatibility class for ChronalIntelligenceEngine.
    Maps to the new WhenEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        try:
            from .when.when_engine import WhenEngine
            self._engine = WhenEngine(config)
            logger.info("ChronalIntelligenceEngine (legacy) initialized - using WhenEngine")
        except ImportError as e:
            logger.error(f"Failed to import WhenEngine: {e}")
            self._engine = None
    
    def analyze_market_data(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze market data using the new WhenEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_market_data(market_data, symbol)
    
    def analyze_temporal_intelligence(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze temporal intelligence using the new WhenEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_temporal_intelligence(market_data, symbol)
    
    def get_dimensional_reading(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> DimensionalReading:
        """Get dimensional reading using the new WhenEngine"""
        if self._engine is None:
            return DimensionalReading(
                dimension="WHEN",
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.0,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Engine not available"]
            )
        return self._engine.get_dimensional_reading(market_data, symbol)
    
    def get_dimension_name(self) -> str:
        """Get dimension name"""
        return "WHEN"
    
    def reset(self) -> None:
        """Reset the engine"""
        if self._engine is not None:
            # Reset logic if needed
            pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Get engine snapshot"""
        return {
            'dimension': 'WHEN',
            'engine_type': 'ChronalIntelligenceEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def update(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Update with new market data"""
        return self.analyze_market_data(market_data, symbol)


class EnhancedFundamentalIntelligenceEngine:
    """
    Legacy compatibility class for EnhancedFundamentalIntelligenceEngine.
    Maps to the new WhyEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        try:
            from .why.why_engine import WhyEngine
            self._engine = WhyEngine(config)
            logger.info("EnhancedFundamentalIntelligenceEngine (legacy) initialized - using WhyEngine")
        except ImportError as e:
            logger.error(f"Failed to import WhyEngine: {e}")
            self._engine = None
    
    def analyze_market_data(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze market data using the new WhyEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_market_data(market_data, symbol)
    
    def analyze_fundamental_intelligence(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze fundamental intelligence using the new WhyEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_fundamental_intelligence(market_data, symbol)
    
    def get_dimensional_reading(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> DimensionalReading:
        """Get dimensional reading using the new WhyEngine"""
        if self._engine is None:
            return DimensionalReading(
                dimension="WHY",
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.0,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Engine not available"]
            )
        return self._engine.get_dimensional_reading(market_data, symbol)
    
    def get_dimension_name(self) -> str:
        """Get dimension name"""
        return "WHY"
    
    def get_diagnostic_information(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        return {
            'dimension': 'WHY',
            'engine_type': 'EnhancedFundamentalIntelligenceEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def reset(self) -> None:
        """Reset the engine"""
        if self._engine is not None:
            # Reset logic if needed
            pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Get engine snapshot"""
        return {
            'dimension': 'WHY',
            'engine_type': 'EnhancedFundamentalIntelligenceEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def update(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Update with new market data"""
        return self.analyze_market_data(market_data, symbol)


class AnomalyIntelligenceEngine:
    """
    Legacy compatibility class for AnomalyIntelligenceEngine.
    Maps to the new AnomalyEngine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        try:
            from .anomaly.anomaly_engine import AnomalyEngine
            self._engine = AnomalyEngine(config)
            logger.info("AnomalyIntelligenceEngine (legacy) initialized - using AnomalyEngine")
        except ImportError as e:
            logger.error(f"Failed to import AnomalyEngine: {e}")
            self._engine = None
    
    def analyze_market_data(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze market data using the new AnomalyEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_market_data(market_data, symbol)
    
    def analyze_anomaly_intelligence(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Analyze anomaly intelligence using the new AnomalyEngine"""
        if self._engine is None:
            return {}
        return self._engine.analyze_anomaly_intelligence(market_data, symbol)
    
    def get_dimensional_reading(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> DimensionalReading:
        """Get dimensional reading using the new AnomalyEngine"""
        if self._engine is None:
            return DimensionalReading(
                dimension="ANOMALY",
                signal_strength=0.0,
                confidence=0.0,
                regime=MarketRegime.UNKNOWN,
                context={},
                data_quality=0.0,
                processing_time_ms=0.0,
                evidence={},
                warnings=["Engine not available"]
            )
        return self._engine.get_dimensional_reading(market_data, symbol)
    
    def get_dimension_name(self) -> str:
        """Get dimension name"""
        return "ANOMALY"
    
    def reset(self) -> None:
        """Reset the engine"""
        if self._engine is not None:
            # Reset logic if needed
            pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Get engine snapshot"""
        return {
            'dimension': 'ANOMALY',
            'engine_type': 'AnomalyIntelligenceEngine',
            'status': 'active' if self._engine is not None else 'inactive'
        }
    
    def update(self, market_data: List[MarketData], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Update with new market data"""
        return self.analyze_market_data(market_data, symbol)


# Legacy enums for backward compatibility
class MarketRegime:
    """Legacy MarketRegime enum"""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class PatternType:
    """Legacy PatternType enum"""
    TRIANGLE = "triangle"
    FLAG = "flag"
    WEDGE = "wedge"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"


class AnomalyType:
    """Legacy AnomalyType enum"""
    PRICE_ANOMALY = "price_anomaly"
    VOLUME_ANOMALY = "volume_anomaly"
    VOLATILITY_ANOMALY = "volatility_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"
    CHAOS_ANOMALY = "chaos_anomaly"


# Additional legacy classes for full compatibility
class MarketRegimeDetector:
    """Legacy MarketRegimeDetector compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            from .when.regime_detection import MarketRegimeDetector as NewDetector
            self._detector = NewDetector(config)
        except ImportError:
            self._detector = None
    
    def detect_market_regime(self, df) -> str:
        if self._detector is None:
            return "unknown"
        return self._detector.detect_market_regime(df)
    
    def update_market_data(self, df) -> Dict[str, Any]:
        if self._detector is None:
            return {}
        return self._detector.update_market_data(df)


class AdvancedPatternRecognition:
    """Legacy AdvancedPatternRecognition compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            from .anomaly.pattern_recognition import AdvancedPatternRecognition as NewPattern
            self._pattern = NewPattern(config)
        except ImportError:
            self._pattern = None
    
    def detect_patterns(self, df) -> List[Dict[str, Any]]:
        if self._pattern is None:
            return []
        return self._pattern.detect_patterns(df)
    
    def update_data(self, df) -> Dict[str, Any]:
        if self._pattern is None:
            return {}
        return self._pattern.update_data(df)


class TemporalAnalyzer:
    """Legacy TemporalAnalyzer compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            from .when.regime_detection import TemporalAnalyzer as NewAnalyzer
            self._analyzer = NewAnalyzer(config)
        except ImportError:
            self._analyzer = None
    
    def update_market_data(self, df) -> Dict[str, Any]:
        if self._analyzer is None:
            return {}
        return self._analyzer.update_market_data(df)
    
    def get_temporal_regime(self, df) -> Dict[str, Any]:
        if self._analyzer is None:
            return {}
        return self._analyzer.get_temporal_regime(df)


class PatternRecognitionDetector:
    """Legacy PatternRecognitionDetector compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        try:
            from .anomaly.pattern_recognition import AdvancedPatternRecognition as NewPattern
            self._pattern = NewPattern(config)
        except ImportError:
            self._pattern = None
    
    def detect_patterns(self, df) -> List[Dict[str, Any]]:
        if self._pattern is None:
            return []
        return self._pattern.detect_patterns(df)
    
    def update_data(self, df) -> Dict[str, Any]:
        if self._pattern is None:
            return {}
        return self._pattern.update_data(df)


# Example usage
if __name__ == "__main__":
    # Test legacy compatibility
    print("Testing legacy compatibility classes...")
    
    # Test engine instantiation
    try:
        engine = InstitutionalMechanicsEngine()
        print("✅ InstitutionalMechanicsEngine: PASS")
    except Exception as e:
        print(f"❌ InstitutionalMechanicsEngine: FAIL - {e}")
    
    try:
        engine = TechnicalRealityEngine()
        print("✅ TechnicalRealityEngine: PASS")
    except Exception as e:
        print(f"❌ TechnicalRealityEngine: FAIL - {e}")
    
    try:
        engine = ChronalIntelligenceEngine()
        print("✅ ChronalIntelligenceEngine: PASS")
    except Exception as e:
        print(f"❌ ChronalIntelligenceEngine: FAIL - {e}")
    
    try:
        engine = EnhancedFundamentalIntelligenceEngine()
        print("✅ EnhancedFundamentalIntelligenceEngine: PASS")
    except Exception as e:
        print(f"❌ EnhancedFundamentalIntelligenceEngine: FAIL - {e}")
    
    try:
        engine = AnomalyIntelligenceEngine()
        print("✅ AnomalyIntelligenceEngine: PASS")
    except Exception as e:
        print(f"❌ AnomalyIntelligenceEngine: FAIL - {e}")
    
    print("Legacy compatibility test completed") 