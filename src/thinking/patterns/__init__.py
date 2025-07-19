"""
EMP Thinking Patterns v1.1

Pattern detection and classification modules for the thinking layer.
These patterns analyze sensory signals to identify market conditions,
anomalies, and regime changes.
"""

from .trend_detector import TrendDetector
from .anomaly_detector import AnomalyDetector
from .regime_classifier import RegimeClassifier
from .cycle_detector import CycleDetector

__all__ = [
    'TrendDetector',
    'AnomalyDetector', 
    'RegimeClassifier',
    'CycleDetector'
] 