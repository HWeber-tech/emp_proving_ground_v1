"""
Analysis package for EMP system.

This package contains:
- Market regime detection
- Pattern recognition
- Advanced market analysis
"""

from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeResult
from .pattern_recognition import AdvancedPatternRecognition, PatternType, PatternResult

__all__ = [
    'MarketRegimeDetector', 'MarketRegime', 'RegimeResult',
    'AdvancedPatternRecognition', 'PatternType', 'PatternResult'
] 